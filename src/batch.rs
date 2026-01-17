use std::cmp;
use std::collections::{HashMap};
use std::ffi::{OsString};
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::rc::{Rc};

use png;

use serde_json as sj;

use crate::data_conv::*;
use crate::shader::*;
use crate::vulkan::*;

// Whether a given dispatch should be considered to
// be in units of threads or workgroups
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchMode {
    // Thread mode. Rounds the final workgroup count up in order
    // to dispatch at least the given number of threads.
    Threads,
    // Workgroup mode. Computed dispatch size is passed through
    // unmodified.
    Workgroups,
}


// 3D Vector component (for working with image sizes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericComponent {
    X,
    Y,
    Z,
}

impl NumericComponent {
    fn parse(val : &sj::Value) -> Result<Self, String> {
        if let sj::Value::String(s) = val {
            if s == "x" { return Ok(Self::X); }
            if s == "y" { return Ok(Self::Y); }
            if s == "z" { return Ok(Self::Z); }
        }

        Err(format!("Invalid component: {val}"))
    }

    fn eval<T : Copy>(&self, (x, y, z) : (T, T, T)) -> T {
        match self {
            NumericComponent::X => x,
            NumericComponent::Y => y,
            NumericComponent::Z => z,
        }
    }
}


// Numeric source 
#[derive(Debug, Clone, PartialEq, Eq)]
enum NumericSource {
    // Constant number
    Constant(i64),
    // Workgroup size of the shader
    WorkgroupSize(NumericComponent),
    // Size of the given buffer, in bytes.
    BufferBytes(String),
    // Number of structures in the given buffer, based on the buffer
    // type declared in the shader. Only works for runtime arrays.
    BufferLength(String),
    // Image extent in given dimension
    ImageExtent(String, NumericComponent),
    // Number of image layers
    ImageLayers(String),
    // Number of image mip levels
    ImageMips(String),
    // Adds two numbers
    Add(Box<NumericSource>, Box<NumericSource>),
    // Subtracts two numbers
    Sub(Box<NumericSource>, Box<NumericSource>),
    // Multiplies two numbers
    Mul(Box<NumericSource>, Box<NumericSource>),
    // Divides two numbers
    Div(Box<NumericSource>, Box<NumericSource>),
    // Minimum of two numbers
    Min(Box<NumericSource>, Box<NumericSource>),
    // Maximum of two numbers
    Max(Box<NumericSource>, Box<NumericSource>),
    // Mip level size
    Mip(Box<NumericSource>, Box<NumericSource>),
}

impl NumericSource {
    fn parse(val : &sj::Value) -> Result<Self, String> {
        if let sj::Value::Number(c) = val {
            return Ok(Self::Constant(c.as_i64()
                .ok_or(format!("Invalid constant: {c}"))?));
        }

        if let sj::Value::Object(o) = val {
            if let Some(wg) = o.get("workgroup-size") {
                return Ok(Self::WorkgroupSize(NumericComponent::parse(wg)?));
            }

            if let Some(sj::Value::String(name)) = o.get("buffer-bytes") {
                return Ok(Self::BufferBytes(name.clone()));
            }

            if let Some(sj::Value::String(name)) = o.get("buffer-elements") {
                return Ok(Self::BufferLength(name.clone()));
            }

            if let Some(sj::Value::String(name)) = o.get("image-width") {
                return Ok(Self::ImageExtent(name.clone(), NumericComponent::X));
            }

            if let Some(sj::Value::String(name)) = o.get("image-height") {
                return Ok(Self::ImageExtent(name.clone(), NumericComponent::Y));
            }

            if let Some(sj::Value::String(name)) = o.get("image-depth") {
                return Ok(Self::ImageExtent(name.clone(), NumericComponent::Z));
            }

            if let Some(sj::Value::String(name)) = o.get("image-layers") {
                return Ok(Self::ImageLayers(name.clone()));
            }

            if let Some(sj::Value::String(name)) = o.get("image-mips") {
                return Ok(Self::ImageMips(name.clone()));
            }

            if let Some(sj::Value::Array(components)) = o.get("add") &&
                    let [a, b] = &components[..] {
                return Ok(Self::Add(
                    Box::new(Self::parse(a)?),
                    Box::new(Self::parse(b)?)));
            }

            if let Some(sj::Value::Array(components)) = o.get("sub") &&
                    let [a, b] = &components[..] {
                return Ok(Self::Sub(
                    Box::new(Self::parse(a)?),
                    Box::new(Self::parse(b)?)));
            }

            if let Some(sj::Value::Array(components)) = o.get("mul") &&
                    let [a, b] = &components[..] {
                return Ok(Self::Mul(
                    Box::new(Self::parse(a)?),
                    Box::new(Self::parse(b)?)));
            }

            if let Some(sj::Value::Array(components)) = o.get("div") &&
                    let [a, b] = &components[..] {
                return Ok(Self::Div(
                    Box::new(Self::parse(a)?),
                    Box::new(Self::parse(b)?)));
            }

            if let Some(sj::Value::Array(components)) = o.get("min") &&
                    let [a, b] = &components[..] {
                return Ok(Self::Min(
                    Box::new(Self::parse(a)?),
                    Box::new(Self::parse(b)?)));
            }

            if let Some(sj::Value::Array(components)) = o.get("max") &&
                    let [a, b] = &components[..] {
                return Ok(Self::Max(
                    Box::new(Self::parse(a)?),
                    Box::new(Self::parse(b)?)));
            }

            if let Some(sj::Value::Array(components)) = o.get("mip") &&
                    let [a, b] = &components[..] {
                return Ok(Self::Mip(
                    Box::new(Self::parse(a)?),
                    Box::new(Self::parse(b)?)));
            }
        }

        Err(format!("Invalid numeric source: {val}"))
    }

    fn eval(&self,
        shader    : &Shader,
        resources : &HashMap<String, Binding>
    ) -> Result<i64, String> {
        match self {
            NumericSource::Constant(x) => Ok(*x),

            NumericSource::WorkgroupSize(component) => {
                let (x, y, z) = shader.workgroup_size();
                Ok(component.eval((x as i64, y as i64, z as i64)))
            },

            NumericSource::BufferBytes(name) => {
                let Some(resource) = resources.get(name) else {
                    return Err(format!("Unknown resource: {name}"));
                };

                match resource {
                    Binding::Null => Ok(0i64),
                    Binding::Buffer(b) => Ok(b.info().size as i64),
                    _ => { return Err(format!("Not a buffer: {name}")); }
                }
            },

            NumericSource::BufferLength(name) => {
                let Some(resource) = resources.get(name) else {
                    return Err(format!("Unknown resource: {name}"));
                };

                let Some(ty) = shader.get_buffer_type(name) else {
                    return Err(format!("Unknown resource: {name}"));
                };

                let Some(member) = ty.members.last() else {
                    return Err(format!("Resource {name} has empty struct"));
                };

                let DataType::RuntimeArray(ty) = &member.data_type else {
                    return Err(format!("Resource {name} does not have unsized array"));
                };

                let struct_size = ty.data_type.size().unwrap() as i64;

                match resource {
                    Binding::Null => Ok(0i64),
                    Binding::Buffer(b) => {
                        let size = b.info().size as i64;
                        let offset = member.offset as i64;
                        
                        Ok((size - offset) / struct_size)
                    },
                    _ => { return Err(format!("Not a buffer: {name}")); }
                }
            },

            NumericSource::ImageExtent(name, component) => {
                let Some(resource) = resources.get(name) else {
                    return Err(format!("Unknown resource: {name}"));
                };

                match resource {
                    Binding::Null => Ok(0i64),
                    Binding::Image(i) => {
                        let (x, y, z) = i.info().extent;
                        Ok(component.eval((x as i64, y as i64, z as i64)))
                    },
                    _ => { return Err(format!("Not an image: {name}")); }
                }
            },

            NumericSource::ImageLayers(name) => {
                let Some(resource) = resources.get(name) else {
                    return Err(format!("Unknown resource: {name}"));
                };

                match resource {
                    Binding::Null => Ok(0i64),
                    Binding::Image(i) => Ok(i.info().layers as i64),
                    _ => { return Err(format!("Not an image: {name}")); }
                }
            },

            NumericSource::ImageMips(name) => {
                let Some(resource) = resources.get(name) else {
                    return Err(format!("Unknown resource: {name}"));
                };

                match resource {
                    Binding::Null => Ok(0i64),
                    Binding::Image(i) => Ok(i.info().mips as i64),
                    _ => { return Err(format!("Not an image: {name}")); }
                }
            },

            NumericSource::Add(a, b) => {
                Ok(a.eval(shader, resources)? + b.eval(shader, resources)?)
            },

            NumericSource::Sub(a, b) => {
                Ok(a.eval(shader, resources)? - b.eval(shader, resources)?)
            },

            NumericSource::Mul(a, b) => {
                Ok(a.eval(shader, resources)? * b.eval(shader, resources)?)
            },

            NumericSource::Div(a, b) => {
                Ok(a.eval(shader, resources)? / b.eval(shader, resources)?)
            },

            NumericSource::Max(a, b) => {
                Ok(cmp::max(a.eval(shader, resources)?, b.eval(shader, resources)?))
            },

            NumericSource::Min(a, b) => {
                Ok(cmp::min(a.eval(shader, resources)?, b.eval(shader, resources)?))
            },

            NumericSource::Mip(a, b) => {
                Ok(cmp::max(1, a.eval(shader, resources)? >> b.eval(shader, resources)?))
            },
        }
    }
}


// Dispatch parameters, to help compute the dispatch size.
#[derive(Debug, Clone)]
struct ShaderDispatch {
    x : (DispatchMode, NumericSource),
    y : (DispatchMode, NumericSource),
    z : (DispatchMode, NumericSource),
}

impl ShaderDispatch {
    // Parses shader dispatch size
    fn parse(val : &sj::Value) -> Result<Self, String> {
        if let sj::Value::Object(obj) = val &&
                let Some(x) = obj.get("x") &&
                let Some(y) = obj.get("y") &&
                let Some(z) = obj.get("z") {
            Ok(Self {
                x : Self::parse_one(x)?,
                y : Self::parse_one(y)?,
                z : Self::parse_one(z)?,
            })
        } else {
            Err(format!("Invalid dispatch object: {val}"))
        }
    }

    fn parse_one(val : &sj::Value) -> Result<(DispatchMode, NumericSource), String> {
        if let Ok(num) = NumericSource::parse(val) {
            return Ok((DispatchMode::Workgroups, num));
        }

        if let Some(obj) = val.get("threads") {
            return Ok((DispatchMode::Threads, NumericSource::parse(obj)?));
        }

        if let Some(obj) = val.get("workgroups") {
            return Ok((DispatchMode::Workgroups, NumericSource::parse(obj)?));
        }

        Err(format!("Invalid dispatch dimension: {val}"))
    }

    // Evaluates workgroup size for a specific shader
    // and specific resources.
    fn eval(&self,
        shader    : &Shader,
        resources : &HashMap<String, Binding>
    ) -> Result<(u32, u32, u32), String> {
        let (x, y, z) = shader.workgroup_size();

        Ok((
            Self::eval_one(shader, resources, &self.x, x)?,
            Self::eval_one(shader, resources, &self.y, y)?,
            Self::eval_one(shader, resources, &self.z, z)?,
        ))
    }

    fn eval_one(
        shader    : &Shader,
        resources : &HashMap<String, Binding>,
        args      : &(DispatchMode, NumericSource),
        wg_size   : u32
    ) -> Result<u32, String> {
        let (mode, source) = args;

        let n = source.eval(shader, resources)?;
        let n : u32 = n.try_into().map_err(|_| "Invalid workgroup count: {n}".to_string())?;

        Ok(match mode {
            DispatchMode::Threads => (n + wg_size - 1) / wg_size,
            DispatchMode::Workgroups => n,
        })
    }
}

impl Default for ShaderDispatch {
    fn default() -> Self {
        // Default to an empty dispatch so that at least the X component
        // must be explicitly specified. Default to workgroup mode.
        Self {
            x : (DispatchMode::Workgroups, NumericSource::Constant(0)),
            y : (DispatchMode::Workgroups, NumericSource::Constant(1)),
            z : (DispatchMode::Workgroups, NumericSource::Constant(1)),
        }
    }
}


// Output buffer size type
#[derive(Debug, Clone)]
enum OutputBufferSizeMode {
    Bytes,
    Elements,
}


// Output buffer properties.
#[derive(Debug, Clone)]
struct OutputBuffer {
    name        : String,
    size_mode   : OutputBufferSizeMode,
    size        : NumericSource,
}

impl OutputBuffer {
    // Parses buffer info from JSON
    fn parse(name : String, val : &sj::Value) -> Result<Self, String> {
        let (mode, size_info) = if let Some(v) = val.get("bytes") {
            (OutputBufferSizeMode::Bytes, v)
        } else if let Some(v) = val.get("elements") {
            (OutputBufferSizeMode::Elements, v)
        } else {
            return Err(format!("Invalid buffer info for {name}: {val}"));
        };

        Ok(Self {
            name      : name,
            size_mode : mode,
            size      : NumericSource::parse(size_info)?,
        })
    }

    // Evaluates output info to buffer description
    fn eval(&self,
        shader    : &Shader,
        resources : &HashMap<String, Binding>
    ) -> Result<BufferInfo, String> {
        let n = self.size.eval(shader, resources)?;

        if n <= 0 {
            return Err(format!("Invalid buffer size: {n}"));
        }

        let ty = shader.get_buffer_type(self.name.as_str())
            .ok_or(format!("Unknown resource: {}", self.name))?;

        let base_size = ty.size();

        let stride = match ty.members.last().map(|m| &m.data_type) {
            Some(DataType::RuntimeArray(t)) => t.data_type.size().unwrap(),
            _ => 0usize
        };

        let bytes = match self.size_mode {
            OutputBufferSizeMode::Bytes     => n as usize,
            OutputBufferSizeMode::Elements  => n as usize * stride + base_size,
        };

        Ok(BufferInfo::default().size(bytes))
    }
}


#[derive(Debug, Clone)]
struct OutputImage {
    dim         : ImageDim,
    format      : ImageFormat,
    extent      : (NumericSource, NumericSource, NumericSource),
    mips        : NumericSource,
    layers      : NumericSource,
}

impl OutputImage {
    // Parses image info from JSON
    fn parse(name : String, val : &sj::Value) -> Result<Self, String> {
        let Some(sj::Value::Number(dim_cnt)) = val.get("dim") else {
            return Err(format!("Missing dim for image {name}: {val}"));
        };

        let dim = match dim_cnt.as_i64() {
            Some(1) => ImageDim::Dim1D,
            Some(2) => ImageDim::Dim2D,
            Some(3) => ImageDim::Dim3D,
            _ => { return Err(format!("Invalid image dimension: {dim_cnt}")) }
        };

        let x = Self::parse_optional(val.get("width"))?;
        let y = Self::parse_optional(val.get("height"))?;
        let z = Self::parse_optional(val.get("depth"))?;

        Ok(Self {
            dim       : dim,
            format    : Self::parse_format(val.get("format"))?,
            extent    : (x, y, z),
            layers    : Self::parse_optional(val.get("layers"))?,
            mips      : Self::parse_optional(val.get("mips"))?,
        })
    }

    fn parse_optional(val : Option<&sj::Value>) -> Result<NumericSource, String> {
        Ok(val.map(|v| NumericSource::parse(v))
            .transpose()?
            .unwrap_or(NumericSource::Constant(1)))
    }

    fn parse_format(val : Option<&sj::Value>) -> Result<ImageFormat, String> {
        let formats = [
            ("R8un", ImageFormat::R8un),
            ("R8sn", ImageFormat::R8sn),
            ("R8ui", ImageFormat::R8ui),
            ("R8si", ImageFormat::R8si),
            ("RG8un", ImageFormat::RG8un),
            ("RG8sn", ImageFormat::RG8sn),
            ("RG8ui", ImageFormat::RG8ui),
            ("RG8si", ImageFormat::RG8si),
            ("RGBA8un", ImageFormat::RGBA8un),
            ("RGBA8sn", ImageFormat::RGBA8sn),
            ("RGBA8ui", ImageFormat::RGBA8ui),
            ("RGBA8si", ImageFormat::RGBA8si),
            ("RGB9E5f", ImageFormat::RGB9E5f),
            ("RGB10A2ui", ImageFormat::RGB10A2ui),
            ("RGB10A2un", ImageFormat::RGB10A2un),
            ("RGB10A2si", ImageFormat::RGB10A2si),
            ("RGB10A2sn", ImageFormat::RGB10A2sn),
            ("R11G11B10f", ImageFormat::R11G11B10f),
            ("R16un", ImageFormat::R16un),
            ("R16sn", ImageFormat::R16sn),
            ("R16ui", ImageFormat::R16ui),
            ("R16si", ImageFormat::R16si),
            ("R16f", ImageFormat::R16f),
            ("RG16un", ImageFormat::RG16un),
            ("RG16sn", ImageFormat::RG16sn),
            ("RG16ui", ImageFormat::RG16ui),
            ("RG16si", ImageFormat::RG16si),
            ("RG16f", ImageFormat::RG16f),
            ("RGBA16un", ImageFormat::RGBA16un),
            ("RGBA16sn", ImageFormat::RGBA16sn),
            ("RGBA16ui", ImageFormat::RGBA16ui),
            ("RGBA16si", ImageFormat::RGBA16si),
            ("RGBA16f", ImageFormat::RGBA16f),
            ("R32ui", ImageFormat::R32ui),
            ("R32si", ImageFormat::R32si),
            ("R32f", ImageFormat::R32f),
            ("RG32ui", ImageFormat::RG32ui),
            ("RG32si", ImageFormat::RG32si),
            ("RG32f", ImageFormat::RG32f),
            ("RGBA32ui", ImageFormat::RGBA32ui),
            ("RGBA32si", ImageFormat::RGBA32si),
            ("R64ui", ImageFormat::R64ui),
            ("R64si", ImageFormat::R64si),
        ];

        let Some(sj::Value::String(fmt)) = val else {
            return Err(format!("Invalid format string: {:?}", val));
        };

        for (s, f) in formats {
            if s == fmt {
                return Ok(f);
            }
        }

        Err(format!("Unknown format: {fmt}"))
    }

    // Evaluates output info to image description
    fn eval(&self,
        shader    : &Shader,
        resources : &HashMap<String, Binding>
    ) -> Result<ImageInfo, String> {
        let (x_src, y_src, z_src) = &self.extent;

        let x = x_src.eval(shader, resources)? as u32;
        let y = y_src.eval(shader, resources)? as u32;
        let z = z_src.eval(shader, resources)? as u32;

        Ok(ImageInfo::default()
            .dim(self.dim)
            .extent((x, y, z))
            .mips(self.mips.eval(shader, resources)? as u32)
            .layers(self.layers.eval(shader, resources)? as u32)
            .format(self.format))
    }
}


// Output resource description. Used to determine properties such
// as the size of a buffer or image based on shader inputs.
#[derive(Debug, Clone)]
enum OutputResourceInfo {
    Buffer(OutputBuffer),
    Image(OutputImage),
}

impl OutputResourceInfo {
    fn parse(val : &sj::Value) -> Result<(String, Self), String> {
        let Some(sj::Value::String(name)) = val.get("name") else {
            return Err(format!("No name in resource description {val}"));
        };

        if let Some(buffer) = val.get("buffer") {
            return Ok((name.clone(), Self::Buffer(OutputBuffer::parse(name.clone(), buffer)?)));
        }

        if let Some(image) = val.get("image") {
            return Ok((name.clone(), Self::Image(OutputImage::parse(name.clone(), image)?)));
        }

        return Err(format!("No 'buffer' or 'image' in resource description {name}: {val}"))
    }
}


// Shader arguments. Can be merged together from different sources.
#[derive(Debug, Default, Clone)]
struct ShaderArgs {
    args : sj::Map<String, sj::Value>,
}

impl ShaderArgs {
    fn parse(args : &sj::Value) -> Result<Self, String> {
        if let sj::Value::Object(args) = args {
            Ok(Self { args : args.clone() })
        } else {
            Err(format!("Invalid shader args: {args}"))
        }
    }

    // Merges another argument set. In case of duplicate keys,
    // the value in the calling map (`self`) will be preserved.
    fn merge(mut self, rhs : &Self) -> Self {
        for (k, v) in rhs.args.iter() {
            if !self.args.contains_key(k) {
                self.args.insert(k.clone(), v.clone());
            }
        }

        self
    }

    // Retrieves arguments as a raw JSON object
    fn into_json(self) -> sj::Value {
        sj::Value::Object(self.args)
    }
}


// Helper trait to parse all objects at once
trait ParseArray : Sized {
    fn parse(val : &sj::Value) -> Result<Self, String>;

    fn parse_all(val : &sj::Value) -> Result<Vec<Self>, String> {
        let sj::Value::Array(passes) = val else {
            return Err("Expected array.".to_string());
        };

        passes.iter().map(|val| Self::parse(val)).collect()
    }
}


// Shader properties
#[derive(Debug, Clone)]
struct ShaderInfo {
    name          : String,
    file          : String,
    dispatch      : ShaderDispatch,
    default_args  : ShaderArgs,
    outputs       : HashMap<String, OutputResourceInfo>,
}

impl ParseArray for ShaderInfo {
    fn parse(val : &sj::Value) -> Result<Self, String> {
        let Some(sj::Value::String(name)) = val.get("name") else {
            return Err("Missing 'name' in shader info.".to_string());
        };

        let Some(sj::Value::String(file)) = val.get("file") else {
            return Err("Missing 'file' in shader info.".to_string());
        };

        let dispatch_info = val.get("dispatch")
            .map(|v| ShaderDispatch::parse(v))
            .transpose()?
            .unwrap_or(ShaderDispatch::default());

        let args = val.get("args")
            .map(|v| ShaderArgs::parse(v))
            .transpose()?
            .unwrap_or(ShaderArgs::default());

        let outputs = val.get("outputs").and_then(|o| o.as_array())
            .ok_or(format!("Missing 'outputs' in shader info: {val}"))?
            .iter().map(|v| OutputResourceInfo::parse(v))
            .collect::<Result<HashMap<_, _>, String>>()?;

        Ok(Self {
            name          : name.clone(),
            file          : file.clone(),
            dispatch      : dispatch_info,
            default_args  : args,
            outputs       : outputs,
        })
    }
}


// Pass info. Establishes resource flow between shaders.
#[derive(Debug, Clone)]
struct PassInfo {
    shader    : String,
    resources : HashMap<String, String>,
    args      : ShaderArgs,
}

impl ParseArray for PassInfo {
    fn parse(val : &sj::Value) -> Result<Self, String> {
        let Some(sj::Value::String(shader)) = val.get("shader") else {
            return Err(format!("Missing 'shader' in pass: {val}"));
        };

        let resources = val.get("resources")
            .ok_or(format!("Missing 'resources' for {shader}: {val}"))?
            .as_object()
            .ok_or(format!("'resources' not an object for {shader}: {val}"))?
            .iter().map(|(k, v)| {
                let v = v.as_str().ok_or(format!("Expected string for {k}"))?;
                Ok::<_, String>((k.clone(), v.to_string()))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        let args = val.get("args")
            .map(|v| ShaderArgs::parse(v))
            .transpose()?
            .unwrap_or(ShaderArgs::default());

        Ok(Self {
            shader    : shader.clone(),
            resources : resources,
            args      : args,
        })
    }
}


// Batch: Establishes a mapping between pass resources and
// actual files on the file system.
#[derive(Debug, Clone)]
struct BatchInfo {
    name    : String,
    inputs  : HashMap<String, String>,
    outputs : HashMap<String, String>,
}

impl ParseArray for BatchInfo {
    fn parse(val : &sj::Value) -> Result<Self, String> {
        let Some(sj::Value::String(name)) = val.get("name") else {
            return Err(format!("Missing 'name' in batch: {val}"));
        };

        let mut batch = BatchInfo {
            name    : name.clone(),
            inputs  : HashMap::new(),
            outputs : HashMap::new(),
        };

        if let Some(sj::Value::Object(inputs)) = val.get("inputs") {
            batch.inputs = Self::parse_map(inputs)?;
        }

        if let Some(sj::Value::Object(outputs)) = val.get("outputs") {
            batch.outputs = Self::parse_map(outputs)?;
        }

        Ok(batch)
    }
}

impl BatchInfo {
    fn parse_map(val : &sj::Map<String, sj::Value>) -> Result<HashMap<String, String>, String> {
        val.iter().map(|(k, v)| {
            let sj::Value::String(v) = v else {
                return Err(format!("Invalid name for {k}: {v}"));
            };

            Ok((k.clone(), v.clone()))
        }).collect::<Result<HashMap<_, _>, _>>()
    }
}


#[derive(Debug, Clone)]
pub struct BatchFile {
    shaders : HashMap<String, ShaderInfo>,
    passes  : Vec<PassInfo>,
    batches : HashMap<String, BatchInfo>,
}

impl BatchFile {
    pub fn parse(val : &sj::Value) -> Result<Self, String> {
        let shaders = ShaderInfo::parse_all(val.get("shaders")
            .unwrap_or(&sj::Value::Array(vec![])))?
            .into_iter()
            .map(|e| (e.name.clone(), e))
            .collect::<HashMap<_, _>>();

        let passes = PassInfo::parse_all(val.get("passes")
            .unwrap_or(&sj::Value::Array(vec![])))?;

        let batches = BatchInfo::parse_all(val.get("batches")
            .unwrap_or(&sj::Value::Array(vec![])))?
            .into_iter()
            .map(|e| (e.name.clone(), e))
            .collect::<HashMap<_, _>>();

        Ok(Self {
            shaders : shaders,
            passes  : passes,
            batches : batches,
        })
    }

    pub fn parse_file(filename : &Path) -> Result<Self, String> {
        let json_string = fs::read_to_string(filename).map_err(
            |e| format!("Failed to open JSON file {}: {e}", filename.display()))?;
        
        let json : sj::Value = sj::from_str(&json_string).map_err(
            |e| format!("Failed to parse JSON file {}: {e}", filename.display()))?;

        Self::parse(&json)
    }

    // Compiles all shaders declared in the file into Vulkan pipelines
    pub fn load_shaders(&self, base_path : &Path, context : &Context) -> Result<ShaderSet, String> {
        Ok(ShaderSet::new(self.shaders.iter().map(|(name, info)| {
            let path = base_path.join(&info.file);
            println!("Loading shader: {}", path.to_str().unwrap());

            let shader = Shader::from_file(&path)?;
            Ok((name.clone(), Pipeline::new(context, shader)?))
        }).collect::<Result<_, String>>()?))
    }

    // Instantiates batch and creates resources.
    pub fn load_batch(&self,
        base_path   : &Path,
        context     : &mut Context,
        shaders     : &ShaderSet,
        batch_name  : &str
    ) -> Result<BatchInstance, String> {
        let batch = self.batches.get(batch_name)
            .ok_or("Unknown batch: {batch_name}")?;

        BatchInstance::new(base_path, context,
            &self.shaders, shaders, &self.passes, &batch)
    }

    // Lists names of available batches
    pub fn list_batches(&self) -> Vec<String> {
        self.batches.iter().map(|(name, _)| name.clone()).collect()
    }
}


// Shader set, used to hold all compiled shaders.
pub struct ShaderSet {
    shaders : HashMap<String, Rc<Pipeline>>,
}

impl ShaderSet {
    fn new(shaders : HashMap<String, Rc<Pipeline>>) -> Self {
        Self {
            shaders : shaders
        }
    }

    pub fn get<'a>(&'a self, name : &str) -> Option<&'a Rc<Pipeline>> {
        self.shaders.get(name)
    }
}


// Generic input resource
trait InputResource {
    fn load(&self, context : &mut Context, shader : &Shader, name : &str) -> Result<Binding, String>;
}


// Null resource
struct InputNullResource { }

impl InputNullResource {
    fn new() -> Self {
        Self { }
    }
}

impl InputResource for InputNullResource {
    fn load(&self, _ : &mut Context, _ : &Shader, _ : &str) -> Result<Binding, String> {
        Ok(Binding::Null)
    }
}


// Untyped buffer read from a binary file.
struct InputBinaryBuffer {
    buffer  : Binding,
}

impl InputBinaryBuffer {
    fn new(path : &Path, context : &mut Context) -> Result<Self, String> {
        println!("Loading raw binary buffer: {}", path.to_str().unwrap());

        let bin = fs::read(&path).map_err(
            |e| format!("Failed to open input resource {}: {e}", path.to_str().unwrap()))?;

        let cpu_buffer_info = BufferInfo::default()
            .size(bin.len())
            .cpu_access(CpuAccess::WRITE);

        let gpu_buffer_info = BufferInfo::default()
            .size(bin.len());

        let cpu_buffer = Buffer::new(context, cpu_buffer_info)?;
        let gpu_buffer = Buffer::new(context, gpu_buffer_info)?;

        let mut cpu_slice = cpu_buffer.get_mapped_mut().unwrap();
        cpu_slice.copy_from_slice(&bin);

        context.copy_buffer(&gpu_buffer, &cpu_buffer)?;

        Ok(Self {
            buffer : Binding::Buffer(gpu_buffer.clone())
        })
    }
}

impl InputResource for InputBinaryBuffer {
    fn load(&self, _ : &mut Context, _ : &Shader, _ : &str) -> Result<Binding, String> {
        Ok(self.buffer.clone())
    }
}


// Typed input buffer read from a JSON file. May be loaded multiple
// times since the actual data layout may differ between shaders
struct InputJsonBuffer {
    json : sj::Value,
}

impl InputJsonBuffer {
    fn new(path : &Path) -> Result<Self, String> {
        println!("Loading JSON buffer: {}", path.to_str().unwrap());

        let json_file = fs::read_to_string(&path).map_err(
            |e| format!("Failed to open input resource {}: {e}", path.to_str().unwrap()))?;

        let json : sj::Value = sj::from_str(&json_file).map_err(
            |e| format!("Failed to parse json file {}: {e}", path.to_str().unwrap()))?;

        Ok(Self { json : json })
    }
}

impl InputResource for InputJsonBuffer {
    fn load(&self, context : &mut Context, shader : &Shader, name : &str) -> Result<Binding, String> {
        let ty = shader.get_buffer_type(name).ok_or(
            format!("Resource {name} not found or not a buffer"))?;

        let buffer_data = from_json(&self.json, ty)?;

        let cpu_buffer_info = BufferInfo::default()
            .size(buffer_data.len())
            .cpu_access(CpuAccess::WRITE);

        let gpu_buffer_info = BufferInfo::default()
            .size(buffer_data.len());

        let cpu_buffer = Buffer::new(context, cpu_buffer_info)?;
        let gpu_buffer = Buffer::new(context, gpu_buffer_info)?;

        let mut cpu_slice = cpu_buffer.get_mapped_mut().unwrap();
        cpu_slice.copy_from_slice(&buffer_data);

        context.copy_buffer(&gpu_buffer, &cpu_buffer)?;

        Ok(Binding::Buffer(gpu_buffer.clone()))
    }
}


// PNG image resource
struct InputPngImage {
    image : Binding,
}

impl InputPngImage {
    fn new(path : &Path, context : &mut Context) -> Result<Self, String> {
        println!("Loading PNG image: {}", path.to_str().unwrap());

        let png_file= fs::File::open(path).map_err(
            |e| format!("Failed to open PNG file {}: {e}", path.to_str().unwrap()))?;

        let mut png_decoder = png::Decoder::new(BufReader::new(png_file));

        png_decoder.set_transformations(
            png::Transformations::EXPAND |
            png::Transformations::ALPHA);

        let mut png_reader = png_decoder.read_info().map_err(
            |e| format!("Failed to read PNG file {}: {e}", path.to_str().unwrap()))?;

        let png_info = png_reader.info();

        let image_info = Self::image_info_from_png(&png_info)?;
        let image = Image::new(context, image_info.clone())?;

        let data_size = png_reader.output_buffer_size().ok_or(
            format!("Failed to query buffer size for {}", path.to_str().unwrap()))?;

        let buffer_info = BufferInfo::default()
            .size(data_size)
            .cpu_access(CpuAccess::WRITE);

        for i in 0..image_info.layers {
            let buffer = Buffer::new(context, buffer_info.clone())?;
            png_reader.next_frame(&mut buffer.get_mapped_mut().unwrap()).map_err(
                |e| format!("Failed to read PNG file {}: {e}", path.to_str().unwrap()))?;

            context.copy_buffer_to_image(&image, 0, i, &buffer)?;
        }

        png_reader.finish().map_err(
            |e| format!("Failed to read PNG file {}: {e}", path.to_str().unwrap()))?;

        Ok(Self {
            image : Binding::Image(image)
        })
    }

    fn image_info_from_png(info : &png::Info<'_>) -> Result<ImageInfo, String> {
        let frame_count = info.animation_control.map(|m| m.num_frames).unwrap_or(1);

        let format = match info.bit_depth {
            png::BitDepth::Eight   => ImageFormat::RGBA8un,
            png::BitDepth::Sixteen => ImageFormat::RGBA16un,
            _ => { return Err(format!("Unsupported bit depth: {}", info.bit_depth as u8)) }
        };

        Ok(ImageInfo::default()
            .format(format)
            .mips(1)
            .layers(frame_count)
            .dim_2d((info.width, info.height)))
    }
}

impl InputResource for InputPngImage {
    fn load(&self, _ : &mut Context, _ : &Shader, _ : &str) -> Result<Binding, String> {
        Ok(self.image.clone())
    }
}


// Generic output resource
trait OutputResource {
    // Copies resource contents into mapped buffer
    fn read_back(&self, context : &mut Context) -> Result<(), String>;

    // Writes mapped buffer contents into file
    fn save_to_file(&self) -> Result<(), String>;
}


// Binary output buffer, simply writes buffer data to a file as-is.
struct OutputBinaryBuffer {
    path          : PathBuf,
    gpu_resource  : Rc<Buffer>,
    cpu_resource  : Rc<Buffer>,
}

impl OutputBinaryBuffer {
    fn new(context : &mut Context, resource : &Binding, path : &Path) -> Result<Self, String> {
        let Binding::Buffer(gpu_buffer) = resource.clone() else {
            return Err(format!("Output resource for {} is not a buffer", path.to_str().unwrap()));
        };

        let buffer_info = gpu_buffer.info().clone().cpu_access(CpuAccess::READ);
        let cpu_buffer = Buffer::new(context, buffer_info)?;

        Ok(Self {
            path          : path.into(),
            gpu_resource  : gpu_buffer,
            cpu_resource  : cpu_buffer,
        })
    }
}

impl OutputResource for OutputBinaryBuffer {
    fn read_back(&self, context : &mut Context) -> Result<(), String> {
        context.copy_buffer(&self.cpu_resource, &self.gpu_resource)
    }

    fn save_to_file(&self) -> Result<(), String> {
        fs::write(&self.path, self.cpu_resource.get_mapped().unwrap()).map_err(
            |e| format!("Failed to write {}: {e}", self.path.to_str().unwrap()))
    }
}


// JSON output buffer, formats output data to a specific data type.
struct OutputJsonBuffer {
    path          : PathBuf,
    data_type     : StructType,
    gpu_resource  : Rc<Buffer>,
    cpu_resource  : Rc<Buffer>,
}

impl OutputJsonBuffer {
    fn new(context : &mut Context, resource : &Binding, data_type : StructType, path : &Path) -> Result<Self, String> {
        let Binding::Buffer(gpu_buffer) = resource.clone() else {
            return Err(format!("Output resource for {} is not a buffer", path.to_str().unwrap()));
        };

        let buffer_info = gpu_buffer.info().clone().cpu_access(CpuAccess::READ);
        let cpu_buffer = Buffer::new(context, buffer_info)?;

        Ok(Self {
            path          : path.into(),
            data_type     : data_type,
            gpu_resource  : gpu_buffer,
            cpu_resource  : cpu_buffer,
        })
    }
}

impl OutputResource for OutputJsonBuffer {
    fn read_back(&self, context : &mut Context) -> Result<(), String> {
        context.copy_buffer(&self.cpu_resource, &self.gpu_resource)
    }

    fn save_to_file(&self) -> Result<(), String> {
        println!("Writing output buffer: {}", self.path.to_str().unwrap());

        let data = self.cpu_resource.get_mapped().unwrap();
        let json = to_json(data, &self.data_type)?;

        let json_text = sj::to_string_pretty(&json).map_err(
            |e| format!("Failed to write JSON text {}: {e}", self.path.to_str().unwrap()))? + "\n";

        fs::write(&self.path, &json_text).map_err(
            |e| format!("Failed to write {}: {e}", self.path.to_str().unwrap()))
    }
}


// PNG output image
struct OutputPngImage {
    path            : PathBuf,
    gpu_resource    : Rc<Image>,
    cpu_resources   : Vec<Rc<Buffer>>,
}

impl OutputPngImage {
    fn new(context : &mut Context, resource : &Binding, path : &Path) -> Result<Self, String> {
        let Binding::Image(gpu_image) = resource.clone() else {
            return Err(format!("Output resource for {} is not an image", path.to_str().unwrap()));
        };

        let (w, h, _) = gpu_image.info().extent;
        let format = gpu_image.info().format;
        let layers = gpu_image.info().layers;

        let buffer_info = BufferInfo::default()
            .size(format.byte_size() * (w * h) as usize)
            .cpu_access(CpuAccess::READ);

        let cpu_buffers = (0..layers).into_iter()
            .map(|_| Buffer::new(context, buffer_info.clone()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            path          : path.into(),
            gpu_resource  : gpu_image,
            cpu_resources : cpu_buffers,
        })
    }

    fn determine_color_type(&self) -> Result<(png::ColorType, png::BitDepth), String> {
        let format = self.gpu_resource.info().format;

        match format {
            ImageFormat::R8un     => Ok((png::ColorType::Grayscale, png::BitDepth::Eight)),
            ImageFormat::RG8un    => Ok((png::ColorType::GrayscaleAlpha, png::BitDepth::Eight)),
            ImageFormat::RGBA8un  => Ok((png::ColorType::Rgba, png::BitDepth::Eight)),
            ImageFormat::R16un    => Ok((png::ColorType::Grayscale, png::BitDepth::Sixteen)),
            ImageFormat::RG16un   => Ok((png::ColorType::GrayscaleAlpha, png::BitDepth::Sixteen)),
            ImageFormat::RGBA16un => Ok((png::ColorType::Rgba, png::BitDepth::Sixteen)),
            _ => Err(format!("Unsupported format for PNG export: {:?}", format))
        }
    }
}

impl OutputResource for OutputPngImage {
    fn read_back(&self, context : &mut Context) -> Result<(), String> {
        for (i, buf) in self.cpu_resources.iter().enumerate() {
            context.copy_image_to_buffer(buf, &self.gpu_resource, 0, i as u32)?;
        }

        Ok(())
    }

    fn save_to_file(&self) -> Result<(), String> {
        println!("Writing output image: {}", self.path.to_str().unwrap());
        
        let file = fs::File::create(&self.path).map_err(
            |e| format!("Failed to create PNG file {}: {e}", self.path.to_str().unwrap()))?;

        let writer = BufWriter::new(file);

        let (w, h, _) = self.gpu_resource.info().extent;
        let (color, depth) = self.determine_color_type()?;

        let mut png_encoder = png::Encoder::new(writer, w, h);
        png_encoder.set_color(color);
        png_encoder.set_depth(depth);

        let mut writer = png_encoder.write_header().map_err(
            |e| format!("Failed to write PNG header {}: {e}", self.path.to_str().unwrap()))?;

        for buf in self.cpu_resources.iter() {
            writer.write_image_data(buf.get_mapped().unwrap()).map_err(
                |e| format!("Failed to write PNG file {}: {e}", self.path.to_str().unwrap()))?;
        }

        writer.finish().map_err(
            |e| format!("Failed to write PNG file {}: {e}", self.path.to_str().unwrap()))
    }
}


// Pass instance, with resource mappings and shader info.
struct PassInstance {
    pipeline        : Rc<Pipeline>,
    args            : sj::Value,
    bindings        : HashMap<String, Binding>,  
    workgroups      : (u32, u32, u32),
}

impl PassInstance {
    pub fn dispatch(&self, context : &mut Context) -> Result<(), String> {
        context.dispatch(
            &self.pipeline,
            self.workgroups,
            &self.args,
            &self.bindings)
    }
}


// Batch instance, containing named resources and implementing
// functionality to store output resources.
#[derive(Default)]
pub struct BatchInstance {
    inputs          : HashMap<String, Box<dyn InputResource>>,
    outputs         : HashMap<String, Box<dyn OutputResource>>,
    pass_instances  : Vec<PassInstance>,
}

impl BatchInstance {
    pub fn dispatch(&self, context : &mut Context) -> Result<(), String> {
        self.pass_instances.iter().fold(Ok(()),
            |accum, pass| accum.and_then(|_| pass.dispatch(context)))
    }

    pub fn store_outputs(&self, context : &mut Context) -> Result<(), String> {
        for (_, resource) in self.outputs.iter() {
            resource.read_back(context)?;
        }

        context.submit()?.wait()?;

        for (_, resource) in self.outputs.iter() {
            resource.save_to_file()?;
        }

        Ok(())
    }
}

impl BatchInstance {
    fn new<'a>(
        base_path   : &Path,
        context     : &mut Context,
        shader_infos: &HashMap<String, ShaderInfo>,
        shaders     : &'a ShaderSet,
        passes      : &[PassInfo],
        batch       : &BatchInfo
    ) -> Result<Self, String> {
        // Load input resources
        let mut instance = BatchInstance::default();
        instance.create_inputs(base_path, context, batch)?;

        // Create passes and remember which shader defined which
        // output resource
        let mut resource_map = HashMap::<String, (&Shader, String, Binding)>::new();

        for pass in passes {
            let shader_info = shader_infos.get(&pass.shader).ok_or(
                format!("Shader not defined: {}", pass.shader))?;

            let pipeline = shaders.get(&pass.shader).unwrap();
            instance.create_pass(context, shader_info, pipeline,
                pass, &mut resource_map)?;
        }

        // Map output resources to batch outputs
        instance.create_outputs(base_path, context, &resource_map, batch)?;

        Ok(instance)
    }

    fn create_inputs(&mut self,
        base_path   : &Path,
        context     : &mut Context,
        batch       : &BatchInfo
    ) -> Result<(), String> {
        for (k, v) in batch.inputs.iter() {
            let path = base_path.join(v);
            let path_ext = path.extension()
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or(OsString::default());

            let resource = {
                if path_ext == "bin" {
                    Ok(Box::new(InputBinaryBuffer::new(&path, context)?) as Box<dyn InputResource>)
                } else if path_ext == "json" {
                    Ok(Box::new(InputJsonBuffer::new(&path)?) as Box<dyn InputResource>)
                } else if path_ext == "png" {
                    Ok(Box::new(InputPngImage::new(&path, context)?) as Box<dyn InputResource>)
                } else {
                    Err(format!("Unrecognized file extension: {}", path.to_str().unwrap()))
                }
            }?;

            self.inputs.insert(k.clone(), resource);
        }

        Ok(())
    }

    fn create_outputs(&mut self,
        base_path   : &Path,
        context     : &mut Context,
        resources   : &HashMap<String, (&Shader, String, Binding)>,
        batch       : &BatchInfo
    ) -> Result<(), String> {
        for (k, v) in batch.outputs.iter() {
            let path = base_path.join(v);
            let path_ext = path.extension()
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or(OsString::default());
            
            let Some((shader, name, resource)) = resources.get(k) else {
                eprintln!("Undefined output resource {k}, skipping.");
                continue;
            };

            let output = {
                if path_ext == "bin" {
                    let buf = OutputBinaryBuffer::new(context, resource, &path)?;
                    Ok(Box::new(buf) as Box<dyn OutputResource>)
                } else if path_ext == "json" {
                    let ty = shader.get_buffer_type(name).ok_or(
                        format!("Invalid buffer resource: for {}: {name}", path.to_str().unwrap()))?;

                    let buf = OutputJsonBuffer::new(context, resource, ty.clone(), &path)?;
                    Ok(Box::new(buf) as Box<dyn OutputResource>)
                } else if path_ext == "png" {
                    let buf = OutputPngImage::new(context, resource, &path)?;
                    Ok(Box::new(buf) as Box<dyn OutputResource>)
                } else {
                    Err(format!("Unrecognized file extension: {}", path.to_str().unwrap()))
                }
            }?;

            self.outputs.insert(k.clone(), output);
        }

        Ok(())
    }

    fn create_pass<'a, 'b>(&'a mut self,
        context     : &mut Context,
        shader_info : &ShaderInfo,
        pipeline    : &'b Rc<Pipeline>,
        pass        : &PassInfo,
        resources   : &mut HashMap<String, (&'b Shader, String, Binding)>
    ) -> Result<&'a PassInstance, String> {
        let shader = pipeline.get_shader();

        // Merge shader arguments
        let shader_args = pass.args.clone().merge(&shader_info.default_args).into_json();

        // Gather all resources, including push data args, and order
        // in such a way that outputs occur last in the list.
        let mut resource_names = shader.resources().iter()
            .map(|(k, _)| k.clone())
            .chain(shader.pointer_args().map(|(k, _)| k.clone()))
            .collect::<Vec<_>>();

        resource_names.sort_by(|a, b| {
            let a_is_output = shader_info.outputs.get(a).is_some();
            let b_is_output = shader_info.outputs.get(b).is_some();

            a_is_output.cmp(&b_is_output)
        });

        let mut shader_resources = HashMap::<String, Binding>::new();

        for k in resource_names {
            if let Some(info) = shader_info.outputs.get(&k) {
                // Create output resources and write them back into the
                // global resource map
                let name = pass.resources.get(&k).unwrap_or(&k);

                let resource = Self::create_output_resource(context,
                    info, shader, &shader_resources)?;
                
                shader_resources.insert(k.clone(), resource.clone());
                resources.insert(name.clone(), (shader, k.clone(), resource));
            } else {
                // Map shader resource to batch resource and look up corresponding
                // input resource, prioritizing outputs from any previous passes
                let name = pass.resources.get(&k).unwrap_or(&k);

                let resource = resources.get(name)
                    .map(|(_, _, r)| Ok::<_, String>(r.clone()))
                    .unwrap_or_else(|| {
                        let input = self.inputs.get(name).ok_or(
                            format!("Undefined input: {name}"))?;
                        
                        input.load(context, shader, &k)
                    })?;

                shader_resources.insert(k.clone(), resource);
            }
        }

        // Determine dispatch size based on the resource
        let workgroups = shader_info.dispatch.eval(shader, &shader_resources)?;

        let pass = PassInstance {
            pipeline    : pipeline.clone(),
            args        : shader_args,
            bindings    : shader_resources,
            workgroups  : workgroups,
        };

        self.pass_instances.push(pass);
        Ok(self.pass_instances.last().unwrap())
    }

    fn create_output_resource(
        context     : &mut Context,
        info        : &OutputResourceInfo,
        shader      : &Shader,
        resources   : &HashMap<String, Binding>
    ) -> Result<Binding, String> {
        match info {
            OutputResourceInfo::Buffer(info) => {
                let buf_info = info.eval(shader, resources)?;
                let buffer = Buffer::new(context, buf_info)?;

                Ok(Binding::Buffer(buffer))
            },

            OutputResourceInfo::Image(info) => {
                let img_info = info.eval(shader, resources)?;
                let image = Image::new(context, img_info)?;
                context.init_image(&image)?;

                Ok(Binding::Image(image))
            },
        }
    }

}

use std::collections::{HashMap};
use std::cmp::{self, Ord, Ordering, PartialOrd};
use std::fs;
use std::ops::{Deref};
use std::path::{Path};

use glslang;

use crate::shader_reflection::*;

// Image dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageDim {
    Dim1D,
    Dim1DArray,
    Dim2D,
    Dim2DArray,
    Dim2DMS,
    Dim2DMSArray,
    Dim3D,
    DimCube,
    DimCubeArray,
}


// Image format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    R8ui,
    R8un,
    R8si,
    R8sn,
    RG8ui,
    RG8un,
    RG8si,
    RG8sn,
    RGBA8ui,
    RGBA8un,
    RGBA8si,
    RGBA8sn,
    RGB9E5f,
    RGB10A2ui,
    RGB10A2un,
    RGB10A2si,
    RGB10A2sn,
    R11G11B10f,
    R16ui,
    R16un,
    R16si,
    R16sn,
    R16f,
    RG16ui,
    RG16un,
    RG16si,
    RG16sn,
    RG16f,
    RGBA16ui,
    RGBA16un,
    RGBA16si,
    RGBA16sn,
    RGBA16f,
    R32ui,
    R32si,
    R32f,
    RG32ui,
    RG32si,
    RG32f,
    RGBA32ui,
    RGBA32si,
    RGBA32f,
    R64ui,
    R64si,
}


// Set and binding index for a shader resource
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceIndex {
    pub set     : u32,
    pub binding : u32,
}

impl ResourceIndex {
    pub fn new(set : u32, binding : u32) -> Self {
        Self {
            set     : set,
            binding : binding,
        }
    }
}

impl Ord for ResourceIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.set.cmp(&other.set)
            .then(self.binding.cmp(&other.binding))
    }
}

impl PartialOrd for ResourceIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
} 


// Resource type and metadata. Enough to determine
// descriptor types as well as view types.
#[derive(Debug, Clone)]
pub enum ResourceType {
    // Sampler descriptor
    Sampler,
    // Image descriptors
    SampledImage(ImageDim),
    StorageImage(ImageDim, Option<ImageFormat>),
    // Buffer descriptors
    UniformBuffer(DataType),
    StorageBuffer(DataType),
}

#[derive(Debug, Clone)]
pub struct Resource {
    pub resource_type     : ResourceType,
    pub binding           : ResourceIndex,
    pub array_size        : u32,
}

impl Resource {
    pub fn new() -> Self {
        Self {
            resource_type     : ResourceType::Sampler,
            binding           : ResourceIndex::new(0, 0),
            array_size        : 1,
        }
    }

    pub fn resource_type(mut self, ty : ResourceType) -> Self {
        self.resource_type = ty;
        self
    }

    pub fn binding(mut self, binding : ResourceIndex) -> Self {
        self.binding = binding;
        self
    }

    pub fn array_size(mut self, count : u32) -> Self {
        self.array_size = count;
        self
    }
}

pub type ResourceMap = HashMap<String, Resource>;


// Scalar data type for shader arguments or buffer data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Sint8,
    Sint16,
    Sint32,
    Sint64,
    Float16,
    Float32,
    Float64,
}

impl ScalarType {
    // Determines size (and alignment) of a scalar type
    pub fn size(&self) -> usize {
        match self {
            ScalarType::Bool => 4usize,

            ScalarType::Uint8 |
            ScalarType::Sint8 => 1usize,

            ScalarType::Uint16 |
            ScalarType::Sint16 |
            ScalarType::Float16 => 2usize,

            ScalarType::Uint32 |
            ScalarType::Sint32 |
            ScalarType::Float32 => 4usize,

            ScalarType::Uint64 |
            ScalarType::Sint64 |
            ScalarType::Float64 => 8usize,
        }
    }
}


// Matrix layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    ColMajor,
    RowMajor,
}


// Matrix type. Takes SPIR-V decoratrions
// for the data layout into account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixType {
    pub size        : (u32, u32),
    pub layout      : MatrixLayout,
    pub stride      : usize,
    pub data_type   : ScalarType,
}

impl MatrixType {
    pub fn size(&self) -> usize {
        let (rows, cols) = self.size;

        let elements = match self.layout {
            MatrixLayout::ColMajor => cols,
            MatrixLayout::RowMajor => rows,
        } as usize;

        self.stride * elements
    }

    pub fn align(&self) -> usize {
        self.data_type.size()
    }
}


// Named struct member entry
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructMember {
    pub name        : String,
    pub offset      : usize,
    pub data_type   : DataType,
}

impl StructMember {
    // Computes offset in struct where this member ends
    // and the next member can begin.
    pub fn end_offset(&self) -> usize {
        let align = self.align();
        let offset = self.offset + self.data_type.size().unwrap_or(0);
        (offset + align - 1) & !(align - 1)
    }

    // Computes required member alignment
    pub fn align(&self) -> usize {
        self.data_type.align()
    }
}


// Struct type, consisting of zero or more members.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructType {
    pub name    : String,
    pub members : Vec<StructMember>,
}

impl StructType {
    pub fn size(&self) -> usize {
        let align = self.align();

        let size = self.members.iter().fold(0usize, |accum, t| {
            cmp::max(accum, t.end_offset())
        });

        (size + align - 1) & !(align - 1)
    }

    pub fn align(&self) -> usize {
        self.members.iter().fold(1usize, |accum, t| {
            cmp::max(accum, t.align())
        })
    }

    pub fn find<'a>(&'a self, name : impl Into<&'a str>) -> Option<&'a StructMember> {
        let name : &'a str = name.into();

        self.members.iter().filter(|m| m.name.as_str() == name).next()
    }

    pub fn get_runtime_array_type<'a>(&'a self) -> Option<&'a DataType> {
        match self.members.last().map(|m| &m.data_type) {
            Some(DataType::RuntimeArray(t)) => Some(&t.data_type),
            _ => None
        }
    }
}


// Sized array type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayType {
    pub size        : usize,
    pub stride      : usize,
    pub data_type   : Box<DataType>,
}

impl ArrayType {
    pub fn size(&self) -> usize {
        self.stride * self.size
    }

    pub fn align(&self) -> usize {
        self.data_type.align()
    }
}


// Unsized array type, used only with buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeArrayType {
    pub stride      : usize,
    pub data_type   : Box<DataType>,
}

impl RuntimeArrayType {
    pub fn align(&self) -> usize {
        self.data_type.align()
    }

    pub fn size(&self, len : usize) -> usize {
        self.stride * len
    }
}


// Data type for shader arguments or buffer data,
// with support for composite types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Scalar(ScalarType),
    Vector(ScalarType, u32),
    Matrix(MatrixType),
    Array(ArrayType),
    RuntimeArray(RuntimeArrayType),
    Struct(StructType),
    Pointer(Box<DataType>),
}

impl DataType {
    pub fn size(&self) -> Option<usize> {
        match self {
            DataType::Scalar(s) => Some(s.size()),
            DataType::Vector(s, n) => Some(s.size() * (*n as usize)),

            DataType::Matrix(t) => Some(t.size()),

            DataType::Array(t) => Some(t.size()),
            DataType::RuntimeArray(_) => None,

            DataType::Struct(t) => Some(t.size()),

            DataType::Pointer(_) => Some(8usize),
        }
    }

    pub fn align(&self) -> usize {
        match self {
            DataType::Scalar(s) |
            DataType::Vector(s, _) => s.size(),

            DataType::Matrix(t) => t.align(),

            DataType::Array(t) => t.align(),
            DataType::RuntimeArray(t) => t.align(),

            DataType::Struct(t) => t.align(),

            DataType::Pointer(_) => 8usize,
        }
    }
}


// Shader object
pub struct Shader {
    code        : Vec<u32>,
    reflection  : Reflection,
}

impl Shader {
    // Raw SPIR-V code that can be passed to Vulkan.
    pub fn code<'a>(&'a self) -> &'a [u32] {
        &self.code
    }

    // Queries workgroup size
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        self.reflection.workgroup_size
    }

    // Named resource bindings
    pub fn resources<'a>(&'a self) -> &'a ResourceMap {
        &self.reflection.resources
    }

    // Named push constants
    pub fn push_constants<'a>(&'a self) -> Option<&'a StructType> {
        self.reflection.push_constants.as_ref()
    }

    // Named push constants with pointer type
    pub fn pointer_args<'a>(&'a self) -> impl Iterator<Item = (&'a String, &'a DataType)> {
        self.push_constants().map(|p| p.members.iter()
            .map(|m| {
                match &m.data_type {
                    DataType::Pointer(t) => Some((&m.name, t.deref())),
                    _ => None
                }
            }).flatten()).into_iter().flatten()
    }

    // Queries data type for a given buffer resource. Returs None
    // if the resource does not exist or is not a buffer.
    pub fn get_buffer_type<'a>(&'a self, name : &'a str) -> Option<&'a StructType> {
        if let Some(res) = self.resources().get(name) {
            match &res.resource_type {
                ResourceType::UniformBuffer(DataType::Struct(t)) |
                ResourceType::StorageBuffer(DataType::Struct(t)) => Some(t),
                _ => None
            }
        } else if let Some(member) = self.push_constants().map(|p| p.find(name)).flatten() {
            match &member.data_type {
                DataType::Pointer(t) => {
                    match t.deref() {
                        DataType::Struct(t) => Some(t),
                        _ => None
                    }
                }
                _ => None
            }
        } else {
            None
        }
    }

    // Create shader object from raw SPIR-V code, and run some basic
    // reflection to gather shader binding metadata
    pub fn from_spv(code : Vec<u32>) -> Result<Self, String> {
        let reflection = Reflection::from_spv(&code)?;

        Ok(Self {
            code        : code,
            reflection  : reflection,
        })
    }

    // Create shader object from byte array
    pub fn from_spv_binary(bytes : &[u8]) -> Result<Self, String> {
        let dword_count = bytes.len() / 4;

        if dword_count * 4 != bytes.len() {
            return Err(format!("SPIR-V binary size {} not a multiple of 4.", bytes.len()));
        }

        let dwords : Vec<_> = (0..dword_count).into_iter().map(|i| {
            u32::from_le_bytes([
                bytes[4 * i + 0], bytes[4 * i + 1],
                bytes[4 * i + 2], bytes[4 * i + 3]])
        }).collect();

        Self::from_spv(dwords)
    }

    // Create shader object either directly from a SPIR-V binary,
    // file, or from a GLSL source file that we can compile.
    pub fn from_file(filename : &Path) -> Result<Self, String> {
        if let Some(ext) = filename.extension().map(|s| s.to_ascii_lowercase()) {
            if ext == "spv" {
                return Self::from_spv_file(filename);
            }

            if ext == "comp" || ext == "glsl" {
                return Self::from_glsl_file(filename);
            }
        }

        Err(format!("Unknown file extension: {}. (supported: .spv, .comp, .glsl)", filename.to_str().unwrap()))
    }

    // Load SPIR-V binary from source file
    pub fn from_spv_file(filename : &Path) -> Result<Self, String> {
        Self::from_spv_binary(&fs::read(filename).map_err(
            |e| format!("Failed to read {}:\n{}.", filename.to_str().unwrap(), e.to_string()))?)
    }

    // Load and compile GLSL source file using reasonable set of default options.
    pub fn from_glsl_file(filename : &Path) -> Result<Self, String> {
        let source = fs::read_to_string(filename).map_err(
            |e| format!("Failed to read {}:\n{}.", filename.to_str().unwrap(), e.to_string()))?;

        let compiler = glslang::Compiler::acquire()
            .ok_or("Failed to acquire glslang instance.")?;

        let code = Self::compile_glsl(&compiler, source).map_err(
            |e| format!("Failed to compile {}:\n{}", filename.to_str().unwrap(), e.to_string()))?;

        Self::from_spv(code)
    }
}

impl Shader {
    fn compile_glsl(compiler : &glslang::Compiler, glsl : String) -> Result<Vec<u32>, glslang::error::GlslangError> {
        let source = glslang::ShaderSource::from(glsl);

        let mut options = glslang::CompilerOptions::default();
        options.target = glslang::Target::Vulkan {
            version : glslang::VulkanVersion::Vulkan1_3,
            spirv_version : glslang::SpirvVersion::SPIRV1_6,
        };

        let input = glslang::ShaderInput::new(&source,
            glslang::ShaderStage::Compute,
            &options,
            None, None)?;

        let shader = compiler.create_shader(input)?;
        shader.compile()
    }
}

use std::collections::{HashMap};
use std::hash::{Hash};

use rspirv::{binary, dr, spirv};

use crate::shader::*;

// Decorations for a given ID or member.
#[derive(Debug, Default, Clone)]
struct SpvDecorations {
    block         : bool,
    buffer_block  : bool,
    non_readable  : bool,
    non_writable  : bool,
    spec_id       : Option<u32>,
    set           : Option<u32>,
    binding       : Option<u32>,
    offset        : Option<u32>,
    array_stride  : Option<u32>,
    matrix_stride : Option<u32>,
    matrix_layout : Option<MatrixLayout>,
}

impl SpvDecorations {
    fn parse(&mut self, kind : spirv::Decoration, operands : &[dr::Operand]) -> binary::ParseAction {
        use dr::Operand::*;

        match kind {
            spirv::Decoration::Block => {
                self.block = true;
                binary::ParseAction::Continue
            },

            spirv::Decoration::BufferBlock => {
                self.buffer_block = true;
                binary::ParseAction::Continue
            },

            spirv::Decoration::Binding => {
                if let LiteralBit32(index) = operands[0] {
                    self.binding = Some(index);
                    binary::ParseAction::Continue
                } else {
                    binary::ParseAction::Stop
                }
            },

            spirv::Decoration::DescriptorSet => {
                if let LiteralBit32(index) = operands[0] {
                    self.set = Some(index);
                    binary::ParseAction::Continue
                } else {
                    binary::ParseAction::Stop
                }
            },

            spirv::Decoration::Offset => {
                if let LiteralBit32(index) = operands[0] {
                    self.offset = Some(index);
                    binary::ParseAction::Continue
                } else {
                    binary::ParseAction::Stop
                }
            },

            spirv::Decoration::ArrayStride => {
                if let LiteralBit32(index) = operands[0] {
                    self.array_stride = Some(index);
                    binary::ParseAction::Continue
                } else {
                    binary::ParseAction::Stop
                }
            },

            spirv::Decoration::MatrixStride => {
                if let LiteralBit32(index) = operands[0] {
                    self.matrix_stride = Some(index);
                    binary::ParseAction::Continue
                } else {
                    binary::ParseAction::Stop
                }
            },

            spirv::Decoration::SpecId => {
                if let LiteralBit32(index) = operands[0] {
                    self.spec_id = Some(index);
                    binary::ParseAction::Continue
                } else {
                    binary::ParseAction::Stop
                }
            },

            spirv::Decoration::ColMajor => {
                self.matrix_layout = Some(MatrixLayout::ColMajor);
                binary::ParseAction::Continue
            },

            spirv::Decoration::RowMajor => {
                self.matrix_layout = Some(MatrixLayout::RowMajor);
                binary::ParseAction::Continue
            },

            spirv::Decoration::NonReadable => {
                self.non_readable = true;
                binary::ParseAction::Continue
            },

            spirv::Decoration::NonWritable => {
                self.non_writable = true;
                binary::ParseAction::Continue
            },

            _ => {
                binary::ParseAction::Continue
            },
        }
    }
}


// SPIR-V image type properties
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpvImageType {
    sampled_type  : spirv::Word,
    dim           : spirv::Dim,
    arrayed       : bool,
    ms            : bool,
    sampled       : bool,
    format        : spirv::ImageFormat,
}


// SPIR-V type info
#[derive(Debug, Clone, PartialEq, Eq)]
enum SpvType {
    Void,
    Bool,
    Int(u32, bool),
    Float(u32),
    Vector(spirv::Word, u32),
    Matrix(spirv::Word, u32),
    Array(spirv::Word, u32),
    RuntimeArray(spirv::Word),
    Pointer(spirv::Word, spirv::StorageClass),
    ForwardPointer(spirv::StorageClass),
    Struct(Vec<spirv::Word>),
    Image(SpvImageType),
    Sampler,
    SampledImage(spirv::Word),
}


// SPIR-V variable info, feeds into reflection info
#[derive(Debug)]
struct SpvVariable {
    storage_class : spirv::StorageClass,
    pointee_type  : spirv::Word,
}


// Constants
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SpvConstant {
    U32(u32),
    I32(i32),

    U64(u64),
    I64(i64),

    Bool(bool),
}


// Entry point declaration
#[derive(Debug, Clone)]
struct SpvEntryPoint {
    id                  : spirv::Word,
    execution_model     : spirv::ExecutionModel,
    vars                : Vec<spirv::Word>,
}


// Execution modes for each entry point
#[derive(Debug, Default, Clone)]
struct SpvExecutionModes {
    local_size          : Option<(u32, u32, u32)>,
    local_size_id       : Option<(spirv::Word, spirv::Word, spirv::Word)>,
}


// Shader reflection parser
#[derive(Debug, Default)]
struct SpvReflector {
    names               : HashMap<spirv::Word, String>,
    decorations         : HashMap<spirv::Word, SpvDecorations>,
    member_names        : HashMap<spirv::Word, Vec<Option<String>>>,
    member_decorations  : HashMap<spirv::Word, Vec<SpvDecorations>>,
    constants           : HashMap<spirv::Word, SpvConstant>,
    types               : HashMap<spirv::Word, SpvType>,
    variables           : HashMap<spirv::Word, SpvVariable>,
    execution_modes     : HashMap<spirv::Word, SpvExecutionModes>,
    entry_points        : HashMap<String, SpvEntryPoint>,
}

impl SpvReflector {
    fn get_reflection(&self) -> Result<Reflection, String> {
        let entry_point = self.entry_points.get("main")
            .ok_or("Entry point 'main' not found".to_string())?;

        if entry_point.execution_model != spirv::ExecutionModel::GLCompute {
            return Err(format!("Shader is not a compute shader, model is {:?}.", entry_point.execution_model));
        }

        Ok(Reflection {
            workgroup_size  : self.get_workgroup_size(entry_point)?,
            resources       : self.get_resource_infos(entry_point)?,
            push_constants  : self.get_push_constants(entry_point)?,
        })
    }

    fn get_workgroup_size(&self, e : &SpvEntryPoint) -> Result<(u32, u32, u32), String> {
        let modes = self.execution_modes.get(&e.id)
            .ok_or(format!("No execution modes for '%{}'", e.id))?;

        if let Some((x, y, z)) = modes.local_size {
            Ok((x, y, z))
        } else if let Some((x_id, y_id, z_id)) = modes.local_size_id {
            let x = self.get_constant_as::<u32>(x_id).ok_or(format!("Invalid constant '%{}'", x_id))?;
            let y = self.get_constant_as::<u32>(y_id).ok_or(format!("Invalid constant '%{}'", y_id))?;
            let z = self.get_constant_as::<u32>(z_id).ok_or(format!("Invalid constant '%{}'", z_id))?;

            Ok((x, y, z))
        } else {
            Err("No workgroup size specified".into())
        }
    }

    fn get_push_constants(&self, e : &SpvEntryPoint) -> Result<Option<StructType>, String> {
        for var_id in &e.vars {
            let var = self.variables.get(&var_id)
                .ok_or(format!("Invalid variable '%{}'", var_id))?;

            if var.storage_class == spirv::StorageClass::PushConstant {
                let Some(DataType::Struct(struct_type)) = self.to_data_type(&var.pointee_type) else {
                    return Err(format!("Invalid var.pointee_type type '%{}'", var.pointee_type));
                };

                return Ok(Some(struct_type));
            }
        }

        Ok(None)
    }

    fn get_resource_infos(&self, e : &SpvEntryPoint) -> Result<ResourceMap, String> {
        let mut map = ResourceMap::new();

        for var_id in &e.vars {
            let var = self.variables.get(&var_id)
                .ok_or(format!("Invalid variable '%{}'", var_id))?;

            let var_ty = self.types.get(&var.pointee_type)
                .ok_or(format!("Invalid type '%{}'", var.pointee_type))?;

            let is_resource = match var.storage_class {
                spirv::StorageClass::Uniform |
                spirv::StorageClass::UniformConstant |
                spirv::StorageClass::StorageBuffer => true,
                _ => false
            };
              
            if !is_resource {
                continue;
            }

            let decorations = self.decorations.get(var_id)
                .ok_or(format!("Missing decorations for '%{}'", var_id))?;

            let name = self.names.get(var_id).cloned()
                .filter(|name| !name.is_empty())
                .or_else(|| self.names.get(&var.pointee_type).cloned())
                .unwrap_or_else(|| format!("_{}", var_id));

            let (ty_id, cnt) = match var_ty {
                SpvType::Array(t, n) => (*t, *n),

                SpvType::Struct(_) |
                SpvType::Image(_) => (var.pointee_type, 1u32),

                _ => { return Err(format!("Unsupported descriptor type: {:#?}", var_ty)); },
            };

            let ty = self.types.get(&ty_id)
                .ok_or(format!("Invalid type ID '%{}'", ty_id))?;

            let resource_type = match ty {
                SpvType::Sampler => ResourceType::Sampler,

                SpvType::Image(t) => self.to_resource_type(t)?,

                SpvType::Struct(_) => {
                    let data_type = self.to_data_type(&ty_id)
                        .ok_or(format!("Invalid data type: {:#?}", ty))?;

                    if var.storage_class == spirv::StorageClass::StorageBuffer || decorations.buffer_block {
                        ResourceType::StorageBuffer(data_type)
                    } else {
                        ResourceType::UniformBuffer(data_type)
                    }
                },

                _ => { return Err(format!("Unsupported resource type: {:#?}", var_ty)); },
            };

            let binding = ResourceIndex::new(
                decorations.set.unwrap_or(0),
                decorations.binding.unwrap_or(0));

            let resource = Resource::new()
                .resource_type(resource_type)
                .binding(binding)
                .array_size(cnt);

            map.insert(name, resource);
        }

        Ok(map)
    }

    fn to_data_type(&self, ty_id : &spirv::Word) -> Option<DataType> {
        let ty = self.types.get(ty_id)?;

        match ty {
            SpvType::Bool => Some(DataType::Scalar(ScalarType::Bool)),

            SpvType::Int( 8, false) => Some(DataType::Scalar(ScalarType::Uint8)),
            SpvType::Int(16, false) => Some(DataType::Scalar(ScalarType::Uint16)),
            SpvType::Int(32, false) => Some(DataType::Scalar(ScalarType::Uint32)),
            SpvType::Int(64, false) => Some(DataType::Scalar(ScalarType::Uint64)),

            SpvType::Int( 8, true) => Some(DataType::Scalar(ScalarType::Sint8)),
            SpvType::Int(16, true) => Some(DataType::Scalar(ScalarType::Sint16)),
            SpvType::Int(32, true) => Some(DataType::Scalar(ScalarType::Sint32)),
            SpvType::Int(64, true) => Some(DataType::Scalar(ScalarType::Sint64)),

            SpvType::Float(16) => Some(DataType::Scalar(ScalarType::Float16)),
            SpvType::Float(32) => Some(DataType::Scalar(ScalarType::Float32)),
            SpvType::Float(64) => Some(DataType::Scalar(ScalarType::Float64)),

            SpvType::Vector(t, n) => {
                if let Some(DataType::Scalar(base)) = self.to_data_type(t) {
                    Some(DataType::Vector(base, *n))
                } else {
                    None
                }
            },

            SpvType::Matrix(t, cols) => {
                if let Some(DataType::Vector(s, rows)) = self.to_data_type(t) {
                    let decorations = self.decorations.get(ty_id)?;
                    
                    let matrix_ty = MatrixType {
                        size      : (rows, *cols),
                        layout    : decorations.matrix_layout.unwrap_or(MatrixLayout::ColMajor),
                        stride    : decorations.matrix_stride? as usize,
                        data_type : s,
                    };

                    Some(DataType::Matrix(matrix_ty))
                } else {
                    None
                }
            },

            SpvType::Array(t, n) => {
                let base = self.to_data_type(t)?;
                let decorations = self.decorations.get(ty_id)?;

                let array_ty = ArrayType {
                    size      : *n as usize,
                    stride    : decorations.array_stride? as usize,
                    data_type : Box::new(base),
                };

                Some(DataType::Array(array_ty))
            },

            SpvType::RuntimeArray(t) => {
                let base = self.to_data_type(t)?;
                let decorations = self.decorations.get(ty_id)?;

                let array_ty = RuntimeArrayType {
                    stride    : decorations.array_stride? as usize,
                    data_type : Box::new(base),
                };

                Some(DataType::RuntimeArray(array_ty))
            },

            SpvType::Pointer(t, class) => {
                if *class != spirv::StorageClass::PhysicalStorageBuffer {
                    return None;
                }

                let base = self.to_data_type(t)?;
                Some(DataType::Pointer(Box::new(base)))
            },

            SpvType::Struct(m) => {
                let name = self.names.get(ty_id).cloned().unwrap_or_else(
                    || format!("_{}", ty_id));

                let members = m.iter().enumerate().map(|(index, member)| {
                    let ty = self.to_data_type(member).ok_or(())?;

                    let name = Self::vec_entry_get(&self.member_names, ty_id, index as u32)
                        .cloned().flatten().unwrap_or_else(|| format!("_m{}", index));

                    let decorations = Self::vec_entry_get(&self.member_decorations, ty_id, index as u32)
                        .ok_or(())?;

                    Ok(StructMember {
                        name      : name,
                        offset    : decorations.offset.ok_or(())? as usize,
                        data_type : ty,
                    })
                }).collect::<Result<Vec<_>, ()>>().ok()?;

                let struct_ty = StructType {
                    name    : name,
                    members : members,
                };

                Some(DataType::Struct(struct_ty))
            }

            _ => None
        }
    }

    fn to_resource_type(&self, ty : &SpvImageType) -> Result<ResourceType, String> {
        if ty.dim == spirv::Dim::DimBuffer {
            Err("Texel buffers not supported".to_string())
        } else {
            let (dim_nor, dim_arr) = match ty.dim {
                spirv::Dim::Dim1D => (Some(ImageDim::Dim1D), Some(ImageDim::Dim1DArray)),
                spirv::Dim::Dim2D => if ty.ms {
                    (Some(ImageDim::Dim2DMS), Some(ImageDim::Dim2DMSArray))
                } else {
                    (Some(ImageDim::Dim2D), Some(ImageDim::Dim2DArray))
                },
                spirv::Dim::Dim3D => (Some(ImageDim::Dim3D), None),
                spirv::Dim::DimCube => (Some(ImageDim::DimCube), Some(ImageDim::DimCubeArray)),
                _ => (None, None)
            };

            let dim = if ty.arrayed { dim_arr } else { dim_nor }
                .ok_or(format!("Invalid dim: {:?}", ty.dim))?;

            if ty.sampled {
                Ok(ResourceType::SampledImage(dim))
            } else {
                Ok(ResourceType::StorageImage(dim, self.to_image_format(ty.format)))
            }
        }
    }

    fn to_image_format(&self, format : spirv::ImageFormat) -> Option<ImageFormat> {
        match format {
            spirv::ImageFormat::R8 => Some(ImageFormat::R8un),
            spirv::ImageFormat::R8Snorm => Some(ImageFormat::R8sn),
            spirv::ImageFormat::R8ui => Some(ImageFormat::R8ui),
            spirv::ImageFormat::R8i => Some(ImageFormat::R8si),

            spirv::ImageFormat::Rg8 => Some(ImageFormat::RG8un),
            spirv::ImageFormat::Rg8Snorm => Some(ImageFormat::RG8sn),
            spirv::ImageFormat::Rg8ui => Some(ImageFormat::RG8ui),
            spirv::ImageFormat::Rg8i => Some(ImageFormat::RG8si),

            spirv::ImageFormat::Rgba8 => Some(ImageFormat::RGBA8un),
            spirv::ImageFormat::Rgba8Snorm => Some(ImageFormat::RGBA8sn),
            spirv::ImageFormat::Rgba8ui => Some(ImageFormat::RGBA8ui),
            spirv::ImageFormat::Rgba8i => Some(ImageFormat::RGBA8si),

            spirv::ImageFormat::R16 => Some(ImageFormat::R16un),
            spirv::ImageFormat::R16Snorm => Some(ImageFormat::R16sn),
            spirv::ImageFormat::R16ui => Some(ImageFormat::R16ui),
            spirv::ImageFormat::R16i => Some(ImageFormat::R16si),
            spirv::ImageFormat::R16f => Some(ImageFormat::R16f),

            spirv::ImageFormat::Rg16 => Some(ImageFormat::RG16un),
            spirv::ImageFormat::Rg16Snorm => Some(ImageFormat::RG16sn),
            spirv::ImageFormat::Rg16ui => Some(ImageFormat::RG16ui),
            spirv::ImageFormat::Rg16i => Some(ImageFormat::RG16si),
            spirv::ImageFormat::Rg16f => Some(ImageFormat::RG16f),

            spirv::ImageFormat::Rgba16 => Some(ImageFormat::RGBA16un),
            spirv::ImageFormat::Rgba16Snorm => Some(ImageFormat::RGBA16sn),
            spirv::ImageFormat::Rgba16ui => Some(ImageFormat::RGBA16ui),
            spirv::ImageFormat::Rgba16i => Some(ImageFormat::RGBA16si),
            spirv::ImageFormat::Rgba16f => Some(ImageFormat::RGBA16f),

            spirv::ImageFormat::R32ui => Some(ImageFormat::R32ui),
            spirv::ImageFormat::R32i => Some(ImageFormat::R32si),
            spirv::ImageFormat::R32f => Some(ImageFormat::R32f),

            spirv::ImageFormat::Rg32ui => Some(ImageFormat::RG32ui),
            spirv::ImageFormat::Rg32i => Some(ImageFormat::RG32si),
            spirv::ImageFormat::Rg32f => Some(ImageFormat::RG32f),

            spirv::ImageFormat::R64ui => Some(ImageFormat::R64ui),
            spirv::ImageFormat::R64i => Some(ImageFormat::R64si),

            spirv::ImageFormat::Rgba32ui => Some(ImageFormat::RGBA32ui),
            spirv::ImageFormat::Rgba32i => Some(ImageFormat::RGBA32si),
            spirv::ImageFormat::Rgba32f => Some(ImageFormat::RGBA32f),

            spirv::ImageFormat::Rgb10A2 => Some(ImageFormat::RGB10A2un),
            spirv::ImageFormat::Rgb10a2ui => Some(ImageFormat::RGB10A2ui),

            spirv::ImageFormat::R11fG11fB10f => Some(ImageFormat::R11G11B10f),

            _ => None,
        }
    }
}

impl binary::Consumer for SpvReflector {
    fn initialize(&mut self) -> binary::ParseAction {
        binary::ParseAction::Continue
    }

    fn finalize(&mut self) -> binary::ParseAction {
        // If we send 'Stop' here, parse returns an error, so don't
        binary::ParseAction::Continue
    }

    fn consume_header(&mut self, _module : dr::ModuleHeader) -> binary::ParseAction {
        binary::ParseAction::Continue
    }

    fn consume_instruction(&mut self, inst : dr::Instruction) -> binary::ParseAction {
        use spirv::Op;
        use dr::Operand::*;

        match inst.class.opcode {
            Op::EntryPoint => {
                let [ExecutionModel(model), IdRef(id), LiteralString(name)] = &inst.operands[..3] else {
                    return binary::ParseAction::Error("Failed to parse entry point.".into());
                };

                let vars : Result<Vec<_>, _> = inst.operands[3..].iter().map(|o| match o {
                    IdRef(id) => Ok(*id),
                    _         => Err(()),
                }).collect();

                let Ok(vars) = vars else {
                    return binary::ParseAction::Error("Failed to parse entry point.".into());
                };

                let entry_point = SpvEntryPoint {
                    id              : *id,
                    execution_model : *model,
                    vars            : vars,
                };

                self.entry_points.insert(name.clone(), entry_point);
            },

            Op::ExecutionMode => {
                let [IdRef(entry_point), ExecutionMode(mode)] = &inst.operands[..2] else {
                    return binary::ParseAction::Error("Failed to parse execution mode.".into());
                };

                if !self.execution_modes.contains_key(entry_point) {
                    self.execution_modes.insert(*entry_point, SpvExecutionModes::default());
                }

                let exec_modes = self.execution_modes.get_mut(entry_point).unwrap();

                match *mode {
                    spirv::ExecutionMode::LocalSize => {
                        let [LiteralBit32(x), LiteralBit32(y), LiteralBit32(z)] = inst.operands[2..] else {
                            return binary::ParseAction::Error("Failed to parse LocalSize execution mode.".into());
                        };

                        exec_modes.local_size = Some((x, y, z));
                    },

                    spirv::ExecutionMode::LocalSizeId => {
                        let [IdRef(x), IdRef(y), IdRef(z)] = inst.operands[2..] else {
                            return binary::ParseAction::Error("Failed to parse LocalSizeId execution mode.".into());
                        };

                        exec_modes.local_size = Some((x, y, z));
                    },

                    _ => { },
                }
            },

            Op::Name => {
                let [IdRef(id), LiteralString(name)] = &inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpName.".into());
                };

                self.names.insert(*id, name.clone());
            },

            Op::MemberName => {
                let [IdRef(id), LiteralBit32(index), LiteralString(name)] = &inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpMemberName.".into());
                };

                *Self::vec_entry_mut(&mut self.member_names, id, *index) = Some(name.clone());
            },

            Op::Decorate => {
                let [IdRef(id), Decoration(kind)] = &inst.operands[..2] else {
                    return binary::ParseAction::Error("Failed to parse OpDecorate.".into());
                };

                if !self.decorations.contains_key(id) {
                    self.decorations.insert(*id, SpvDecorations::default());
                }

                self.decorations.get_mut(id).unwrap().parse(*kind, &inst.operands[2..]);
            },

            Op::MemberDecorate => {
                let [IdRef(id), LiteralBit32(index), Decoration(kind)] = &inst.operands[..3] else {
                    return binary::ParseAction::Error("Failed to parse OpMemberDecorate.".into());
                };

                Self::vec_entry_mut(&mut self.member_decorations, &id, *index)
                    .parse(*kind, &inst.operands[3..]);
            },

            Op::TypeVoid => {
                let id = inst.result_id.unwrap();
                self.types.insert(id, SpvType::Void);
            },

            Op::TypeBool => {
                let id = inst.result_id.unwrap();
                self.types.insert(id, SpvType::Bool);
            },

            Op::TypeInt => {
                let id = inst.result_id.unwrap();

                let [LiteralBit32(width), LiteralBit32(sign)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeInt.".into());
                };

                self.types.insert(id, SpvType::Int(width, sign != 0));
            },

            Op::TypeFloat => {
                let id = inst.result_id.unwrap();

                let [LiteralBit32(width)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeFloat.".into());
                };

                self.types.insert(id, SpvType::Float(width));
            },

            Op::TypeVector => {
                let id = inst.result_id.unwrap();

                let [IdRef(base_id), LiteralBit32(size)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeVector.".into());
                };

                self.types.insert(id, SpvType::Vector(base_id, size));
            },

            Op::TypeMatrix => {
                let id = inst.result_id.unwrap();

                let [IdRef(base_id), LiteralBit32(size)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeMatrix.".into());
                };

                self.types.insert(id, SpvType::Matrix(base_id, size));
            },

            Op::TypeArray => {
                let id = inst.result_id.unwrap();

                let [IdRef(base_id), IdRef(size_id)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeArray.".into());
                };

                let Some(size) = self.get_constant_as::<u32>(size_id) else {
                    return binary::ParseAction::Error("Failed to parse OpTypeArray.".into());
                };

                self.types.insert(id, SpvType::Array(base_id, size));
            },

            Op::TypeRuntimeArray => {
                let id = inst.result_id.unwrap();

                let [IdRef(base_id)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeRuntimeArray.".into());
                };

                self.types.insert(id, SpvType::RuntimeArray(base_id));
            },

            Op::TypeStruct => {
                let id = inst.result_id.unwrap();

                let members : Result<Vec<_>, _> = inst.operands.iter().map(|ty| {
                    let &IdRef(ty_id) = ty else {
                        return Err(format!("Invalid type ID"));
                    };

                    Ok(ty_id)
                }).collect();

                match members {
                    Ok(m) => { self.types.insert(id, SpvType::Struct(m)); },
                    Err(e) => { return binary::ParseAction::Error(format!("Failed to parse OpTypeStruct: {}", e).into()); },
                }
            },

            Op::TypePointer => {
                let id = inst.result_id.unwrap();

                let [StorageClass(class), IdRef(base)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypePointer.".into());
                };

                self.types.insert(id, SpvType::Pointer(base, class));
            },

            Op::TypeForwardPointer => {
                let [IdRef(id), StorageClass(class)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeForwardPointer.".into());
                };

                self.types.insert(id, SpvType::ForwardPointer(class));
            },

            Op::TypeImage => {
                let id = inst.result_id.unwrap();

                let [
                    IdRef(sampled_type),
                    Dim(dim),
                    LiteralBit32(_),
                    LiteralBit32(arrayed),
                    LiteralBit32(ms),
                    LiteralBit32(sampled),
                    ImageFormat(format)
                ] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeImage.".into());
                };

                let image_type = SpvImageType {
                    sampled_type  : sampled_type,
                    dim           : dim,
                    arrayed       : arrayed != 0,
                    ms            : ms != 0,
                    sampled       : sampled < 2,
                    format        : format,
                };

                self.types.insert(id, SpvType::Image(image_type));
            },

            Op::TypeSampler => {
                let id = inst.result_id.unwrap();
                self.types.insert(id, SpvType::Sampler);
            },

            Op::TypeSampledImage => {
                let id = inst.result_id.unwrap();

                let [IdRef(base)] = inst.operands[..] else {
                    return binary::ParseAction::Error("Failed to parse OpTypeSampledImage.".into());
                };

                self.types.insert(id, SpvType::SampledImage(base));
            },

            Op::ConstantTrue | Op::SpecConstantTrue => {
                self.constants.insert(inst.result_id.unwrap(), SpvConstant::Bool(true));
            },

            Op::ConstantFalse | Op::SpecConstantFalse => {
                self.constants.insert(inst.result_id.unwrap(), SpvConstant::Bool(false));
            },

            Op::Constant | Op::SpecConstant => {
                let id = inst.result_id.unwrap();
                let ty_id = inst.result_type.unwrap();

                if let Some(ty) = self.types.get(&ty_id) {
                    let constant = match ty {
                        SpvType::Int(8, false) |
                        SpvType::Int(16, false) |
                        SpvType::Int(32, false) => {
                            if let LiteralBit32(n) = inst.operands[0] {
                                Some(SpvConstant::U32(n))
                            } else {
                                None
                            }
                        },

                        SpvType::Int(64, false) => {
                            if let LiteralBit64(n) = inst.operands[0] {
                                Some(SpvConstant::U64(n))
                            } else {
                                None
                            }
                        },

                        SpvType::Int(8, true) |
                        SpvType::Int(16, true) |
                        SpvType::Int(32, true) => {
                            if let LiteralBit32(n) = inst.operands[0] {
                                Some(SpvConstant::I32(n as i32))
                            } else {
                                None
                            }
                        },

                        SpvType::Int(64, true) => {
                            if let LiteralBit64(n) = inst.operands[0] {
                                Some(SpvConstant::I64(n as i64))
                            } else {
                                None
                            }
                        },

                        _ => None,
                    };

                    if let Some(constant) = constant {
                        self.constants.insert(id, constant);
                    }
                }
            },

            Op::Variable => {
                let id = inst.result_id.unwrap();
                let ptr_type_id = inst.result_type.unwrap();

                let StorageClass(class) = inst.operands[0] else {
                    return binary::ParseAction::Error("Failed to parse OpVariable.".into());
                };

                let Some(&SpvType::Pointer(var_type_id, _)) = self.types.get(&ptr_type_id) else {
                    return binary::ParseAction::Error("Failed to parse OpVariable.".into());
                };

                let variable = SpvVariable {
                    pointee_type  : var_type_id,
                    storage_class : class,
                };

                self.variables.insert(id, variable);
            },

            _ => { },
        }

        binary::ParseAction::Continue
    }
}

impl SpvReflector {
    fn vec_entry_mut<'a, K, V>(map : &'a mut HashMap<K, Vec<V>>, k : &K, index : u32) -> &'a mut V
    where K : Eq + Hash + Clone,
          V : Default {
        if !map.contains_key(k) {
            map.insert(k.clone(), vec![]);
        }

        let v = map.get_mut(k).unwrap();

        while v.len() <= index as usize {
            v.push(V::default());
        }

        &mut v[index as usize]
    }

    fn vec_entry_get<'a, K, V>(map : &'a HashMap<K, Vec<V>>, k : &K, index : u32) -> Option<&'a V>
    where K : Eq + Hash {
        map.get(k).map(|v| v.get(index as usize)).flatten()
    }

    fn get_constant_as<T>(&self, id : u32) -> Option<T>
    where T : TryFrom<u32> + TryFrom<u64> + TryFrom<i32> + TryFrom<i64> {
        self.constants.get(&id).map(|&c| match c {
            SpvConstant::U32(n) => n.try_into().ok(),
            SpvConstant::U64(n) => n.try_into().ok(),
            SpvConstant::I32(n) => n.try_into().ok(),
            SpvConstant::I64(n) => n.try_into().ok(),
            _ => None,
        }).flatten()
    }
}


// Shader reflection data
#[derive(Debug, Default, Clone)]
pub struct Reflection {
    pub workgroup_size  : (u32, u32, u32),
    pub resources       : ResourceMap,
    pub push_constants  : Option<StructType>,
}

impl Reflection {
    pub fn from_spv(code : &[u32]) -> Result<Self, String> {
        let mut reflector = SpvReflector::default();
        
        binary::parse_words(code, &mut reflector).map_err(
            |e| format!("Failed to parse SPIR_V: {}", e.to_string()))?;

        reflector.get_reflection()
    }
}

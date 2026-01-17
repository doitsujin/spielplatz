use serde_json as sj;

use crate::shader::*;

pub fn f32_to_f16(n : f32) -> u16 {
    let n_bits = n.to_bits();

    let exp32 = (n_bits & 0x7f800000) >> 23;
    let frc32 = n_bits & 0x007fffff;

    let sgn16 = (n_bits & 0x80000000) >> 16;

    let exp16 : u32;
    let mut frc16 : u32;

    if exp32 > 142 {
        if exp32 == 0xff {
            // Infinity or NaN, preserve.
            exp16 = 0x1f;
            frc16 = frc32 >> 13;

            if frc32 != 0 {
                frc16 |= 0x200;
            }
        } else {
            // Regular number that is larger what we can represent
            // with f16, return maximum representable number.
            exp16 = 0x1e;
            frc16 = 0x3ff;
        }
    } else if exp32 < 113 {
        if exp32 >= 103 {
            // Number can be represented as denorm
            exp16 = 0;
            frc16 = (0x0400 | (frc32 >> 13)) >> (113 - exp32);
        } else {
            // Number too small to be represented
            exp16 = 0;
            frc16 = 0;
        }
    } else {
        // Regular number
        exp16 = exp32 - 112;
        frc16 = frc32 >> 13;
    }

    (sgn16 | (exp16 << 10) | frc16) as u16
}

pub fn f16_to_f32(n : u16) -> f32 {
    let n = n as u32;

    let exp16 = (n & 0x7c00) >> 10;
    let frc16 = n & 0x03ff;

    let sgn32 = (n & 0x8000) << 16;

    let exp32 : u32;
    let mut frc32 : u32;

    if exp16 == 0 {
        if frc16 == 0 {
            exp32 = 0;
            frc32 = 0;
        } else {
            // Denorm in 16-bit, but we can represent these
            // natively in 32-bit by adjusting the exponent.
            let bit = frc16.leading_zeros();

            exp32 = 135 - bit;
            frc32 = (frc16 << (bit - 9)) & 0x007fffff;
        }
    } else if exp16 == 0x1f {
        // Infinity or NaN, preserve semantic meaning. 
        exp32 = 0xff;
        frc32 = frc16 << 13;

        if frc16 != 0 {
            frc32 |= 0x400000;
        }
    } else {
        // Regular finite number, adjust the exponent as
        // necessary and shift the fractional part.
        exp32 = exp16 + 112;
        frc32 = frc16 << 13;
    }

    f32::from_bits(sgn32 | (exp32 << 23) | frc32)
}


fn encode_bytes<const N : usize>(dst : &mut [u8], src : [u8; N]) -> Result<(), String> {
    if N <= dst.len() {
        dst[0..N].copy_from_slice(&src);
        Ok(())
    } else {
        Err("Slice too small".to_string())
    }
}


fn from_json_scalar(dst : &mut [u8], j : &sj::Value, ty : ScalarType) -> Result<(), String> {
    let sj::Value::Number(value) = j else {
        return Err("Exected Number".to_string());
    };

    let (f, u, i) = if let Some(v) = value.as_f64() {
        (v, v as u64, v as i64)
    } else if let Some(v) = value.as_u64() {
        (v as f64, v, v as i64)
    } else if let Some(v) = value.as_i64() {
        (v as f64, v as u64, v)
    } else {
        return Err("Unknown value type".to_string())
    };

    match ty {
        ScalarType::Bool    => encode_bytes(dst,
            if f != 0.0 || u != 0 { 1u32 } else { 0u32 }.to_le_bytes()),

        ScalarType::Uint8   => encode_bytes(dst, [u as u8]),
        ScalarType::Uint16  => encode_bytes(dst, (u as u16).to_le_bytes()),
        ScalarType::Uint32  => encode_bytes(dst, (u as u32).to_le_bytes()),
        ScalarType::Uint64  => encode_bytes(dst, u.to_le_bytes()),

        ScalarType::Sint8   => encode_bytes(dst, [i as u8]),
        ScalarType::Sint16  => encode_bytes(dst, (i as i16).to_le_bytes()),
        ScalarType::Sint32  => encode_bytes(dst, (i as i32).to_le_bytes()),
        ScalarType::Sint64  => encode_bytes(dst, i.to_le_bytes()),

        ScalarType::Float16 => encode_bytes(dst, f32_to_f16(f as f32).to_le_bytes()),
        ScalarType::Float32 => encode_bytes(dst, (f as f32).to_le_bytes()),
        ScalarType::Float64 => encode_bytes(dst, f.to_le_bytes()),
    }
}

fn from_json_vector(dst : &mut [u8], j : &sj::Value, ty : ScalarType, n : usize) -> Result<(), String> {
    let sj::Value::Array(value) = j else {
        return Err("Exected Array".to_string());
    };

    if n > value.len() {
        return Err(format!("Exected Array of size {}, got {}", n, value.len()));
    }

    let size = ty.size();

    for i in 0..n {
        from_json_scalar(&mut dst[i * size..], &value[i], ty)?;
    }

    Ok(())
}

fn from_json_matrix(dst : &mut [u8], j : &sj::Value, ty : &MatrixType) -> Result<(), String> {
    let sj::Value::Array(value) = j else {
        return Err("Exected Array".to_string());
    };

    let (rows, cols) = ty.size;
    let n = (rows * cols) as usize;

    if n > value.len() {
        return Err(format!("Exected Array of size {}, got {}", n, value.len()));
    }

    let size = ty.data_type.size();

    for i in 0..n {
        // The json layout is trivially row-major
        let src_index = match ty.layout {
            MatrixLayout::RowMajor => i,
            MatrixLayout::ColMajor => {
                let row = i % (rows as usize);
                let col = i / (rows as usize);

                row * (cols as usize) + col
            },
        };

        from_json_scalar(&mut dst[i * size..], &value[src_index], ty.data_type)?;
    }

    Ok(())
}

fn from_json_array(dst : &mut [u8], j : &sj::Value, ty : &ArrayType) -> Result<(), String> {
    let sj::Value::Array(value) = j else {
        return Err("Exected Array".to_string());
    };

    if ty.size > value.len() {
        return Err(format!("Exected Array of size {}, got {}", ty.size, value.len()));
    }

    let size = ty.data_type.size().unwrap();
    let stride = ty.stride;

    for i in 0..ty.size {
        let a = i * stride;
        let b = i * stride + size;

        from_json_into(&mut dst[a..b], &value[i], &ty.data_type)?;
    }

    Ok(())
}

fn from_json_runtime_array(dst : &mut [u8], j : &sj::Value, ty : &RuntimeArrayType) -> Result<(), String> {
    let sj::Value::Array(value) = j else {
        return Err("Exected Array".to_string());
    };

    // Accept any size including 0.
    let n = value.len();

    let size = ty.data_type.size().unwrap();
    let stride = ty.stride;

    for i in 0..n {
        let a = i * stride;
        let b = i * stride + size;

        from_json_into(&mut dst[a..b], &value[i], &ty.data_type)?;
    }

    Ok(())
}

fn from_json_struct(dst : &mut [u8], j : &sj::Value, ty : &StructType) -> Result<(), String> {
    let sj::Value::Object(value) = j else {
        return Err("Expected struct".to_string());
    };

    for m in &ty.members {
        // Skip pointers, those are resolved later
        if let DataType::Pointer(_) = &m.data_type {
            continue;
        }

        let Some(v) = value.get(&m.name) else {
            return Err(format!("No data for {}::{}", ty.name, m.name));
        };

        from_json_into(&mut dst[m.offset..], v, &m.data_type)?;
    }

    Ok(())
}

fn from_json_pointer(_ : &mut [u8], j : &sj::Value) -> Result<(), String> {
    match j {
        sj::Value::String(_) => Ok(()),
        _ => Err("Expected String for pointer type".to_string()),
    }
}

pub fn from_json_into(dst : &mut [u8], j : &sj::Value, ty : &DataType) -> Result<(), String> {
    match ty {
        DataType::Scalar(s) => from_json_scalar(dst, j, *s),
        DataType::Vector(s, n) => from_json_vector(dst, j, *s, *n as usize),
        DataType::Matrix(t) => from_json_matrix(dst, j, t),
        DataType::Array(t) => from_json_array(dst, j, t),
        DataType::Struct(t) => from_json_struct(dst, j, t),
        DataType::RuntimeArray(t) => from_json_runtime_array(dst, j, t),
        DataType::Pointer(_) => from_json_pointer(dst, j),
    }
}

pub fn from_json(j : &sj::Value, ty : &StructType) -> Result<Vec<u8>, String> {
    let sj::Value::Object(value) = j else {
        return Err("Expected struct".to_string());
    };

    // Allocate extra size for a runtime array if necessary
    let mut size = ty.size();

    for m in &ty.members {
        if let DataType::RuntimeArray(t) = &m.data_type {
            let Some(sj::Value::Array(array)) = value.get(&m.name) else {
                return Err(format!("No data for {}::{}", ty.name, m.name));
            };

            size += t.size(array.len());
        }
    }

    // Allocate storage and serialize data into it
    let mut data : Vec<u8> = vec![0; size];
    from_json_struct(&mut data, j, ty)?;

    Ok(data)
}


fn decode_bytes<const N : usize>(src : &[u8]) -> Result<[u8; N], String> {
    if N <= src.len() {
        let mut data = [0u8; N];
        data.copy_from_slice(&src[0..N]);
        Ok(data)
    } else {
        Err(format!("Failed to read {N} bytes from buffer"))
    }
}

fn to_json_scalar(src : &[u8], ty : ScalarType) -> Result<sj::Value, String> {
    if src.len() < ty.size() {
        return Err(format!("Type {:?} requires {} bytes, got {}", ty, ty.size(), src.len()))
    }

    let v = match ty {
        ScalarType::Bool => {
            let v = decode_bytes::<4>(src)?;
            sj::Value::Bool(u32::from_le_bytes(v) != 0)
        },

        ScalarType::Uint8 => {
            sj::Value::Number(sj::Number::from_u128(src[0] as u128).unwrap())
        },

        ScalarType::Uint16 => {
            let v = decode_bytes::<2>(src)?;
            sj::Value::Number(sj::Number::from_u128(u16::from_le_bytes(v) as u128).unwrap())
        },

        ScalarType::Uint32 => {
            let v = decode_bytes::<4>(src)?;
            sj::Value::Number(sj::Number::from_u128(u32::from_le_bytes(v) as u128).unwrap())
        },

        ScalarType::Uint64 => {
            let v = decode_bytes::<8>(src)?;
            sj::Value::Number(sj::Number::from_u128(u64::from_le_bytes(v) as u128).unwrap())
        },

        ScalarType::Sint8 => {
            sj::Value::Number(sj::Number::from_i128((src[0] as i8) as i128).unwrap())
        },

        ScalarType::Sint16 => {
            let v = decode_bytes::<2>(src)?;
            sj::Value::Number(sj::Number::from_i128(i16::from_le_bytes(v) as i128).unwrap())
        },

        ScalarType::Sint32 => {
            let v = decode_bytes::<4>(src)?;
            sj::Value::Number(sj::Number::from_i128(i32::from_le_bytes(v) as i128).unwrap())
        },

        ScalarType::Sint64 => {
            let v = decode_bytes::<8>(src)?;
            sj::Value::Number(sj::Number::from_i128(i64::from_le_bytes(v) as i128).unwrap())
        },

        ScalarType::Float16 => {
            let v = decode_bytes::<2>(src)?;
            sj::Value::Number(sj::Number::from_f64(f16_to_f32(u16::from_le_bytes(v)) as f64).unwrap())
        },

        ScalarType::Float32 => {
            let v = decode_bytes::<4>(src)?;
            sj::Value::Number(sj::Number::from_f64(f32::from_le_bytes(v) as f64).unwrap())
        },

        ScalarType::Float64 => {
            let v = decode_bytes::<8>(src)?;
            sj::Value::Number(sj::Number::from_f64(f64::from_le_bytes(v)).unwrap())
        },
    };

    Ok(v)
}

fn to_json_vector(src : &[u8], ty : ScalarType, n : usize) -> Result<sj::Value, String> {
    let size = ty.size();
    let mut array = Vec::<sj::Value>::with_capacity(n);

    for i in 0..n {
        let a = i * size;
        array.push(to_json_scalar(&src[a..], ty).map_err(
            |e| format!("{e}\n  For type: {:#?}", ty))?)
    }

    Ok(sj::Value::Array(array))
}

fn to_json_matrix(src : &[u8], ty : &MatrixType) -> Result<sj::Value, String> {
    let size = ty.size();
    let (rows, cols) = ty.size;

    let mut array = Vec::<sj::Value>::with_capacity((rows * cols) as usize);

    for r in 0..rows {
        for c in 0..cols {
            let offset = match ty.layout {
                MatrixLayout::RowMajor => (r as usize) * ty.stride + (c as usize) * size,
                MatrixLayout::ColMajor => (c as usize) * ty.stride + (r as usize) * size,
            };

            array.push(to_json_scalar(&src[offset..], ty.data_type).map_err(
                |e| format!("{e}\n  For type: {:#?}", ty))?)
        }
    }

    Ok(sj::Value::Array(array))
}

fn to_json_array(src : &[u8], ty : &ArrayType) -> Result<sj::Value, String> {
    let size = ty.data_type.size().unwrap();
    let stride = ty.stride;

    let mut array = Vec::<sj::Value>::with_capacity(ty.size);

    for i in 0..ty.size {
        let a = i * stride;
        let b = i * stride + size;

        array.push(to_json_from(&src[a..b], &ty.data_type).map_err(
            |e| format!("{e}\n  At array index {i}"))?)
    }

    Ok(sj::Value::Array(array))
}

fn to_json_runtime_array(src : &[u8], ty : &RuntimeArrayType) -> Result<sj::Value, String> {
    let size = ty.data_type.size().unwrap();
    let stride = ty.stride;

    let n = src.len() / stride;

    let mut array = Vec::<sj::Value>::with_capacity(n);

    for i in 0..n {
        let a = i * stride;
        let b = i * stride + size;

        array.push(to_json_from(&src[a..b], &ty.data_type).map_err(
            |e| format!("{e}\n  At array index {i}"))?)
    }

    Ok(sj::Value::Array(array))
}

fn to_json_struct(src : &[u8], ty : &StructType) -> Result<sj::Value, String> {
    let mut map = sj::Map::<String, sj::Value>::new();

    for m in &ty.members {
        map.insert(m.name.clone(),
            to_json_from(&src[m.offset..], &m.data_type).map_err(
                |e| format!("{e}\n  For {}::{} (offset {})", ty.name, m.name, m.offset))?);
    }

    Ok(sj::Value::Object(map))
}

pub fn to_json_from(src : &[u8], ty : &DataType) -> Result<sj::Value, String> {
    match ty {
        DataType::Scalar(t) => to_json_scalar(src, *t),
        DataType::Vector(t, n) => to_json_vector(src, *t, *n as usize),
        DataType::Matrix(t) => to_json_matrix(src, t),
        DataType::Array(t) => to_json_array(src, t),
        DataType::RuntimeArray(t) => to_json_runtime_array(src, t),
        DataType::Struct(t) => to_json_struct(src, t),
        DataType::Pointer(_) => Err("Cannot output pointers".into()),
    }
}

pub fn to_json(src : &[u8], ty : &StructType) -> Result<sj::Value, String> {
    to_json_struct(src, ty)
}

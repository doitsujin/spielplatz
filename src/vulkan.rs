use std::collections::{HashMap};
use std::cell::{RefCell};
use std::cmp;
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::rc::{Rc};
use std::slice;

use bitflags::bitflags;

use ash::{self, ext, khr, vk};
use sdl3;

use serde_json as sj;

use crate::data_conv::*;
use crate::shader::*;

fn make_string_from_vk<const N : usize>(s : &[c_char; N]) -> String {
    make_cstring_from_vk(s).into_string().unwrap()
}

fn make_cstring_from_vk<const N : usize>(s : &[c_char; N]) -> CString {
    let mut bytes : [u8; N] = [0; N];

    for (i, &c) in s.iter().enumerate() {
        bytes[i] = c as u8;
    }

    CStr::from_bytes_until_nul(&bytes).unwrap().into()
}


// Resource binding info
#[derive(Clone)]
pub enum Binding {
    Null,
    Buffer(Rc<Buffer>),
    Image(Rc<Image>),
    Sampler(Rc<Sampler>),
}


// Determines Vulkan descriptor type for resource type
impl From<ResourceType> for vk::DescriptorType {
    fn from(t : ResourceType) -> Self {
        match t {
            ResourceType::Sampler => vk::DescriptorType::SAMPLER,
            ResourceType::SampledImage(_) => vk::DescriptorType::SAMPLED_IMAGE,
            ResourceType::StorageImage(_, _) => vk::DescriptorType::STORAGE_IMAGE,
            ResourceType::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            ResourceType::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}


// Determines Vulkan image type from image dimension
impl From<ImageDim> for vk::ImageType {
    fn from(t : ImageDim) -> Self {
        match t {
            ImageDim::Dim1D |
            ImageDim::Dim1DArray => vk::ImageType::TYPE_1D,

            ImageDim::Dim2D |
            ImageDim::Dim2DArray |
            ImageDim::Dim2DMS |
            ImageDim::Dim2DMSArray |
            ImageDim::DimCube |
            ImageDim::DimCubeArray => vk::ImageType::TYPE_2D,

            ImageDim::Dim3D => vk::ImageType::TYPE_3D,
        }
    }
}


// Determines Vulkan image view type from image dimension
impl From<ImageDim> for vk::ImageViewType {
    fn from(t : ImageDim) -> Self {
        match t {
            ImageDim::Dim1D => vk::ImageViewType::TYPE_1D,
            ImageDim::Dim1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
            ImageDim::Dim2D |
            ImageDim::Dim2DMS => vk::ImageViewType::TYPE_2D,
            ImageDim::Dim2DArray |
            ImageDim::Dim2DMSArray => vk::ImageViewType::TYPE_2D_ARRAY,
            ImageDim::DimCube => vk::ImageViewType::CUBE,
            ImageDim::DimCubeArray => vk::ImageViewType::CUBE_ARRAY,
            ImageDim::Dim3D => vk::ImageViewType::TYPE_3D,
        }
    }
}


// Translates built-in format to Vulkan format
impl From<ImageFormat> for vk::Format {
    fn from(t : ImageFormat) -> Self {
        match t {
            ImageFormat::R8ui          => vk::Format::R8_UINT,
            ImageFormat::R8un          => vk::Format::R8_UNORM,
            ImageFormat::R8si          => vk::Format::R8_SINT,
            ImageFormat::R8sn          => vk::Format::R8_SNORM,
            ImageFormat::RG8ui         => vk::Format::R8G8_UINT,
            ImageFormat::RG8un         => vk::Format::R8G8_UNORM,
            ImageFormat::RG8si         => vk::Format::R8G8_SINT,
            ImageFormat::RG8sn         => vk::Format::R8G8_SNORM,
            ImageFormat::RGBA8ui       => vk::Format::R8G8B8A8_UINT,
            ImageFormat::RGBA8un       => vk::Format::R8G8B8A8_UNORM,
            ImageFormat::RGBA8si       => vk::Format::R8G8B8A8_SINT,
            ImageFormat::RGBA8sn       => vk::Format::R8G8B8A8_SNORM,
            ImageFormat::RGB9E5f       => vk::Format::E5B9G9R9_UFLOAT_PACK32,
            ImageFormat::RGB10A2ui     => vk::Format::A2B10G10R10_UINT_PACK32,
            ImageFormat::RGB10A2un     => vk::Format::A2B10G10R10_UNORM_PACK32,
            ImageFormat::RGB10A2si     => vk::Format::A2B10G10R10_SINT_PACK32,
            ImageFormat::RGB10A2sn     => vk::Format::A2B10G10R10_SNORM_PACK32,
            ImageFormat::R11G11B10f    => vk::Format::B10G11R11_UFLOAT_PACK32,
            ImageFormat::R16ui         => vk::Format::R16_UINT,
            ImageFormat::R16un         => vk::Format::R16_UNORM,
            ImageFormat::R16si         => vk::Format::R16_SINT,
            ImageFormat::R16sn         => vk::Format::R16_SNORM,
            ImageFormat::R16f          => vk::Format::R16_SFLOAT,
            ImageFormat::RG16ui        => vk::Format::R16G16_UINT,
            ImageFormat::RG16un        => vk::Format::R16G16_UNORM,
            ImageFormat::RG16si        => vk::Format::R16G16_SINT,
            ImageFormat::RG16sn        => vk::Format::R16G16_SNORM,
            ImageFormat::RG16f         => vk::Format::R16G16_SFLOAT,
            ImageFormat::RGBA16ui      => vk::Format::R16G16B16A16_UINT,
            ImageFormat::RGBA16un      => vk::Format::R16G16B16A16_UNORM,
            ImageFormat::RGBA16si      => vk::Format::R16G16B16A16_SINT,
            ImageFormat::RGBA16sn      => vk::Format::R16G16B16A16_SNORM,
            ImageFormat::RGBA16f       => vk::Format::R16G16B16A16_SFLOAT,
            ImageFormat::R32ui         => vk::Format::R32_UINT,
            ImageFormat::R32si         => vk::Format::R32_SINT,
            ImageFormat::R32f          => vk::Format::R32_SFLOAT,
            ImageFormat::RG32ui        => vk::Format::R32G32_UINT,
            ImageFormat::RG32si        => vk::Format::R32G32_SINT,
            ImageFormat::RG32f         => vk::Format::R32G32_SFLOAT,
            ImageFormat::RGBA32ui      => vk::Format::R32G32B32A32_UINT,
            ImageFormat::RGBA32si      => vk::Format::R32G32B32A32_SINT,
            ImageFormat::RGBA32f       => vk::Format::R32G32B32A32_SFLOAT,
            ImageFormat::R64ui         => vk::Format::R64_UINT,
            ImageFormat::R64si         => vk::Format::R64_SINT,
        }
    }
}


// CPU access flags for a resource
bitflags! {
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct CpuAccess : u32 {
        const READ  = 0x1;
        const WRITE = 0x2;
    }
}

impl CpuAccess {
    fn get_memory_property_flags(&self) -> vk::MemoryPropertyFlags {
        if self.contains(CpuAccess::READ) {
            vk::MemoryPropertyFlags::HOST_VISIBLE |
            vk::MemoryPropertyFlags::HOST_CACHED
        } else if self.contains(CpuAccess::WRITE) {
            vk::MemoryPropertyFlags::HOST_VISIBLE |
            vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        }
    }
}


// Vulkan device creation parameters
#[derive(Debug, Clone)]
pub struct VulkanInfo {
    pub adapter_index : u32,
    pub debug         : bool,
}

impl VulkanInfo {
    pub fn new() -> Self {
        Self {
            adapter_index : 0,
            debug         : false,
        }
    }

    pub fn adapter_index(mut self, index : u32) -> Self {
        self.adapter_index = index;
        self
    }

    pub fn debug(mut self, enable : bool) -> Self {
        self.debug = enable;
        self
    }
}


// Enabled or supported Vulkan extensions
#[derive(Debug, Default)]
struct Extensions {
    khr_swapchain     : bool,
    khr_maintenance5  : bool,
}

impl Extensions {
    fn from_list(extensions : &[vk::ExtensionProperties]) -> Self {
        let mut result = Self::default();

        let ext_mapping = result.get_mapping();

        for (name, flag) in ext_mapping {
            *flag = extensions.iter().find(|info| {
                let s = make_cstring_from_vk(&info.extension_name);
                s.as_c_str() == name
            }).is_some()
        }

        result
    }

    fn get_names(&mut self) -> Vec<*const c_char> {
        self.get_mapping().into_iter().map(|(name, flag)| {
            if *flag {
                Some(name.as_ptr())
            } else {
                None
            }
        }).flatten().collect()
    }

    fn get_mapping<'a>(&'a mut self) -> Vec<(&'static CStr, &'a mut bool)> {
        vec![
            (khr::swapchain::NAME,    &mut self.khr_swapchain),
            (khr::maintenance5::NAME, &mut self.khr_maintenance5),
        ]
    }
}


// Enabled or supported device features
#[derive(Debug, Default, Clone)]
struct Features<'a> {
    core              : vk::PhysicalDeviceFeatures2<'a>,
    vk11              : vk::PhysicalDeviceVulkan11Features<'a>,
    vk12              : vk::PhysicalDeviceVulkan12Features<'a>,
    vk13              : vk::PhysicalDeviceVulkan13Features<'a>,
    khr_maintenance5  : vk::PhysicalDeviceMaintenance5FeaturesKHR<'a>,
}

macro_rules! copy_members {
    ($dst:expr, $src:expr, [$($feature:ident,)*]) => {
        {
            $(
                $dst.$feature = $src.$feature.clone();
            )*
        }
    }
}


// Device properties
#[derive(Debug, Default, Clone)]
struct Properties {
    core              : vk::PhysicalDeviceProperties2<'static>,
    vk11              : vk::PhysicalDeviceVulkan11Properties<'static>,
    vk12              : vk::PhysicalDeviceVulkan12Properties<'static>,
    vk13              : vk::PhysicalDeviceVulkan13Properties<'static>,
}


// Reference-counted SDL subsystem
struct SdlInstance {
    instance    : sdl3::Sdl,
    video       : sdl3::VideoSubsystem,
    event       : sdl3::EventSubsystem,
    event_pump  : sdl3::EventPump,
    window      : sdl3::video::Window,
}

impl SdlInstance {
    fn new() -> Result<Self, String> {
        let instance = sdl3::init().map_err(|e| e.to_string())?;
        let video = instance.video().map_err(|e| e.to_string())?;
        let event = instance.event().map_err(|e| e.to_string())?;
        let event_pump = instance.event_pump().map_err(|e| e.to_string())?;

        video.vulkan_load_library_default().map_err(|e| e.to_string())?;

        let window = sdl3::video::WindowBuilder::new(&video, "Spielplatz", 1280, 720)
            .position_centered()
            .high_pixel_density()
            .vulkan()
            .hidden()
            .build()
            .map_err(|e| e.to_string())?;

        Ok(Self {
            instance    : instance,
            video       : video,
            event       : event,
            event_pump  : event_pump,
            window      : window,
        })
    }

    fn get_vulkan_entry_point(&self) -> Result<ash::StaticFn, String> {
        let entry_point = self.video.vulkan_get_proc_address_function()
            .ok_or("Could not locate vkGetInstanceProcAddr")?;

        Ok(ash::StaticFn {
            // SAFETY: We verified that the entry point is non-null, but types
            // aren't compatible between SDL and Ash so we need to cast.
            get_instance_proc_addr : unsafe { mem::transmute(entry_point ) }
        })
    }

    fn get_vulkan_extensions(&self) -> Result<Vec<CString>, String> {
        let extensions = self.window.vulkan_instance_extensions().map_err(|e| e.to_string())?;

        Ok(extensions.into_iter().map(
            |ext| CString::new(ext).unwrap()).collect::<Vec<_>>())
    }
}


// Reference-counted Vulkan instance
struct VulkanInstance {
    _sdl            : SdlInstance,
    _vk_entry       : ash::Entry,
    vk_instance     : ash::Instance,
    ext_debug_utils : Option<ext::debug_utils::Instance>,
    vk_messenger    : vk::DebugUtilsMessengerEXT,
}

impl VulkanInstance {
    fn new(info : &VulkanInfo) -> Result<Self, String> {
        let sdl = SdlInstance::new()?;

        let vk_entry = unsafe {
            // SAFETY: Vulkan library and entry point are alive and well
            ash::Entry::from_static_fn(sdl.get_vulkan_entry_point()?)
        };

        let vk_version = unsafe {
            vk_entry.try_enumerate_instance_version().map_err(
                |e| format!("Failed to query Vulkan version: {}", e.to_string()))?
                .unwrap_or(vk::make_api_version(0, 1, 0, 0))
        };
        
        if vk_version < vk::make_api_version(0, 1, 1, 0) {
            return Err("Vulkan instance version 1.1 not available.".into());
        }

        let (layer_infos, extension_infos) = unsafe {
            (vk_entry.enumerate_instance_layer_properties().map_err(
                |e| format!("Failed to query instance layers: {}", e.to_string()))?,
             vk_entry.enumerate_instance_extension_properties(None).map_err(
                |e| format!("Failed to query instance layers: {}", e.to_string()))?)
        };

        let mut layers : Vec<CString> = vec![];

        let mut extensions : Vec<CString> = vec![
            khr::get_surface_capabilities2::NAME.into(),
        ];

        if info.debug {
            layers.push(c"VK_LAYER_KHRONOS_validation".into());
            extensions.push(ext::debug_utils::NAME.into());
        }

        for ext in sdl.get_vulkan_extensions()?.into_iter() {
            if extensions.iter().find(|&e| *e == ext).is_none() {
                extensions.push(ext);
            }
        }

        for l in layers.iter() {
            if layer_infos.iter().find(|info| {
                *l == make_cstring_from_vk(&info.layer_name)
            }).is_none() {
                return Err(format!("Layer not supported: {}", l.clone().into_string().unwrap()));
            }
        }

        for e in extensions.iter() {
            if extension_infos.iter().find(|info| {
                *e == make_cstring_from_vk(&info.extension_name)
            }).is_none() {
                return Err(format!("Extension not supported: {}", e.clone().into_string().unwrap()));
            }
        }

        let layer_names : Vec<*const c_char> = layers.iter()
            .map(|s| s.as_c_str().as_ptr()).collect();

        let extension_names : Vec<*const c_char> = extensions.iter()
            .map(|s| s.as_c_str().as_ptr()).collect();

        let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION |
                vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
            .pfn_user_callback(Some(Self::debug_callback));

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Spielplatz")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names);

        if info.debug {
            instance_info = instance_info.push_next(&mut messenger_info);
        }

        let vk_instance = unsafe {
            vk_entry.create_instance(&instance_info, None).map_err(
                |e| format!("Failed to create Vulkan instance: {}", e.to_string()))?
        };

        let ext_debug_utils = if info.debug {
            Some(ext::debug_utils::Instance::new(&vk_entry, &vk_instance))
        } else {
            None
        };

        let mut vk_messenger = vk::DebugUtilsMessengerEXT::null();

        if let Some(ext_debug_utils) = &ext_debug_utils {
            let status = unsafe {
                ext_debug_utils.create_debug_utils_messenger(&messenger_info, None).map_err(
                    |e| format!("Failed to create debug utils messenger: {}", e.to_string()))
            };

            match status {
                Ok(object)  => { vk_messenger = object; },
                Err(e)      => { eprintln!("{}", e.to_string()); },
            }
        };

        Ok(Self {
            _sdl            : sdl,
            _vk_entry       : vk_entry,
            vk_instance     : vk_instance,
            ext_debug_utils : ext_debug_utils,
            vk_messenger    : vk_messenger,
        })
    }

    fn get<'a>(&'a self) -> &'a ash::Instance {
        &self.vk_instance
    }
}

impl VulkanInstance {
    unsafe extern "system" fn debug_callback(
        _severity   : vk::DebugUtilsMessageSeverityFlagsEXT,
        _type_flags : vk::DebugUtilsMessageTypeFlagsEXT,
        data        : *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data  : *mut c_void) -> u32 {
        let data = unsafe {
            // SAFETY: We kinda have to assume that Vulkan gives us
            // valid pointers to work with if they are not null
            if data.is_null() {
                return vk::FALSE;
            }
            
            *data
        };

        let prefix = unsafe {
            if !data.p_message_id_name.is_null() {
                format!("{}", CStr::from_ptr(data.p_message_id_name).to_str().unwrap().to_string())
            } else {
                format!("{}", data.message_id_number)
            }
        };

        let message = unsafe {
            if !data.p_message.is_null() {
                format!("{}", CStr::from_ptr(data.p_message).to_str().unwrap().to_string())
            } else {
                "".to_string()
            }
        };

        eprintln!("{}: {}", prefix, message);
        vk::FALSE
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            if let Some(ext_debug_utils) = self.ext_debug_utils.take() {
                ext_debug_utils.destroy_debug_utils_messenger(self.vk_messenger, None);
            }

            self.vk_instance.destroy_instance(None);
        }
    }
}


// Reference-counted Vulkan device
struct VulkanDevice {
    instance      : Rc<VulkanInstance>,
    properties    : Properties,
    features      : Features<'static>,
    memory        : vk::PhysicalDeviceMemoryProperties,
    queue_family  : u32,
    vk_device     : ash::Device,
    vk_queue      : vk::Queue,
}

impl VulkanDevice {
    fn new(instance : &Rc<VulkanInstance>, info : &VulkanInfo) -> Result<Self, String> {
        let vk = instance.get();

        let adapters = unsafe {
            vk.enumerate_physical_devices().map_err(
                |e| format!("Failed to query adapters: {}", e.to_string()))?
        };

        let adapter = adapters.get(info.adapter_index as usize)
            .ok_or(format!("Invalid adapter index: {}", info.adapter_index))?.clone();

        let extension_infos = unsafe {
            vk.enumerate_device_extension_properties(adapter).map_err(
                |e| format!("Failed to query device extensions: {}", e.to_string()))?
        };

        let mut extensions = Extensions::from_list(&extension_infos);

        let mut supported_features = Features::default();
        let mut supported_properties = Properties::default();

        let mut core_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut supported_features.vk11)
            .push_next(&mut supported_features.vk12)
            .push_next(&mut supported_features.vk13)
            .push_next(&mut supported_features.khr_maintenance5);

        let mut core_properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut supported_properties.vk11)
            .push_next(&mut supported_properties.vk12)
            .push_next(&mut supported_properties.vk13);

        {
            unsafe {
                vk.get_physical_device_features2(adapter, &mut core_features);
                vk.get_physical_device_properties2(adapter, &mut core_properties);
            }
        }

        let memory = unsafe {
            vk.get_physical_device_memory_properties(adapter)
        };

        // The borrow checker is going to defeat any attempt at copying feature structs
        // around that have their pNext chains set up already, so we have to enable all
        // features we want by hand, and also copy all device properties that we care
        // about.
        let mut features = Features::default();

        copy_members!(&mut features.core.features, &core_features.features, [
            robust_buffer_access,
            sampler_anisotropy,
            shader_image_gather_extended,
            shader_storage_image_extended_formats,
            shader_storage_image_multisample,
            shader_storage_image_read_without_format,
            shader_storage_image_write_without_format,
            shader_uniform_buffer_array_dynamic_indexing,
            shader_sampled_image_array_dynamic_indexing,
            shader_storage_buffer_array_dynamic_indexing,
            shader_storage_image_array_dynamic_indexing,
            shader_float64,
            shader_int64,
            shader_int16,
        ]);

        copy_members!(&mut features.vk11, &supported_features.vk11, [
            storage_buffer16_bit_access,
            uniform_and_storage_buffer16_bit_access,
            storage_push_constant16,
            storage_input_output16,
        ]);

        copy_members!(&mut features.vk12, &supported_features.vk12, [
            storage_buffer8_bit_access,
            uniform_and_storage_buffer8_bit_access,
            storage_push_constant8,
            shader_buffer_int64_atomics,
            shader_shared_int64_atomics,
            shader_float16,
            shader_int8,
            descriptor_indexing,
            shader_uniform_buffer_array_non_uniform_indexing,
            shader_sampled_image_array_non_uniform_indexing,
            shader_storage_buffer_array_non_uniform_indexing,
            shader_storage_image_array_non_uniform_indexing,
            shader_input_attachment_array_non_uniform_indexing,
            scalar_block_layout,
            uniform_buffer_standard_layout,
            shader_subgroup_extended_types,
            timeline_semaphore,
            buffer_device_address,
            vulkan_memory_model,
            subgroup_broadcast_dynamic_id,
        ]);

        copy_members!(&mut features.vk13, &supported_features.vk13, [
            robust_image_access,
            subgroup_size_control,
            compute_full_subgroups,
            synchronization2,
            shader_zero_initialize_workgroup_memory,
            shader_integer_dot_product,
            maintenance4,
        ]);

        copy_members!(&mut features.khr_maintenance5, &supported_features.khr_maintenance5, [
            maintenance5,
        ]);

        // Clone that list so we can set up features to enable.
        let mut enabled_features = features.clone();

        enabled_features.core = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enabled_features.vk11)
            .push_next(&mut enabled_features.vk12)
            .push_next(&mut enabled_features.vk13)
            .push_next(&mut enabled_features.khr_maintenance5);

        // Copy known properties
        let mut properties = Properties::default();

        copy_members!(&mut properties.core, &core_properties, [ properties, ]);

        copy_members!(&mut properties.vk11, &supported_properties.vk11, [
            device_uuid,
            driver_uuid,
            device_luid,
            device_luid_valid,
            subgroup_size,
            subgroup_supported_stages,
            subgroup_supported_operations,
            subgroup_quad_operations_in_all_stages,
            max_memory_allocation_size,
        ]);

        copy_members!(&mut properties.vk12, &supported_properties.vk12, [
            driver_id,
            driver_name,
            driver_info,
            conformance_version,
            denorm_behavior_independence,
            rounding_mode_independence,
            shader_signed_zero_inf_nan_preserve_float16,
            shader_signed_zero_inf_nan_preserve_float32,
            shader_signed_zero_inf_nan_preserve_float64,
            shader_denorm_preserve_float16,
            shader_denorm_preserve_float32,
            shader_denorm_preserve_float64,
            shader_denorm_flush_to_zero_float16,
            shader_denorm_flush_to_zero_float32,
            shader_denorm_flush_to_zero_float64,
            shader_rounding_mode_rte_float16,
            shader_rounding_mode_rte_float32,
            shader_rounding_mode_rte_float64,
            shader_rounding_mode_rtz_float16,
            shader_rounding_mode_rtz_float32,
            shader_rounding_mode_rtz_float64,
        ]);

        copy_members!(&mut properties.vk13, &supported_properties.vk13, [
            min_subgroup_size,
            max_subgroup_size,
            max_compute_workgroup_subgroups,
            required_subgroup_size_stages,
            max_buffer_size,
        ]);

        let extension_names = extensions.get_names();

        // Pick first available queue, this is probably going to
        // be a graphics queue but that is fine.
        let queue_properties = unsafe {
            vk.get_physical_device_queue_family_properties(adapter)
        };

        let queue_index = queue_properties.iter().enumerate()
            .filter(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32)
            .next()
            .ok_or("No compute-capable queue found".to_string())?;

        let queue_prios = [1.0f32];

        let queue_info = [
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_index)
                .queue_priorities(&queue_prios)
        ];

        let device_info = vk::DeviceCreateInfo::default()
            .push_next(&mut enabled_features.core)
            .enabled_extension_names(&extension_names)
            .queue_create_infos(&queue_info);

        let vk_device = unsafe {
            vk.create_device(adapter, &device_info, None).map_err(
                |e| format!("Failed to create Vulkan device: {}", e.to_string()))?
        };

        let vk_queue = unsafe {
            vk_device.get_device_queue(queue_index, 0)
        };

        println!("Using device {}", make_string_from_vk(
            &properties.core.properties.device_name));

        Ok(Self {
            instance      : instance.clone(),
            features      : features,
            properties    : properties,
            memory        : memory,
            queue_family  : queue_index,
            vk_device     : vk_device,
            vk_queue      : vk_queue,
        })
    }

    fn get<'a>(&'a self) -> &'a ash::Device {
        &self.vk_device
    }

    fn find_memory_type(&self, mut mask : u32, cpu_access : CpuAccess) -> Option<u32> {
        let required_flags = cpu_access.get_memory_property_flags();

        while mask != 0 {
            let index = mask.trailing_zeros();

            if self.memory.memory_types[index as usize].property_flags.contains(required_flags) {
                return Some(index);
            }

            mask &= mask - 1;
        }

        None
    }
}

impl Drop for VulkanDevice{
    fn drop(&mut self) {
        unsafe {
            self.vk_device.destroy_device(None);
        }
    }
}


// Trait for lifetime-tracked objects
trait Trackable { }


// Descriptor set layout with metadata
struct PipelineBindings {
    device            : Rc<VulkanDevice>,
    resources         : Vec<(String, Resource)>,
    vk_set_layout     : vk::DescriptorSetLayout,
}

impl PipelineBindings {
    fn new(device : &Rc<VulkanDevice>, resources : Vec<(String, Resource)>) -> Result<Self, String> {
        let vk = device.get();

        let bindings : Vec<_> = resources.iter().map(
            |(_, r)| { Self::make_binding_info(r) }).collect();

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        let vk_set_layout = unsafe {
            vk.create_descriptor_set_layout(&create_info, None).map_err(|e|
                format!("Failed to create descriptor set layout: {}", e.to_string()))?
        };

        Ok(Self {
            device        : device.clone(),
            resources     : resources,
            vk_set_layout : vk_set_layout,
        })
    }

    fn make_binding_info(r : &Resource) -> vk::DescriptorSetLayoutBinding<'_> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(r.binding.binding)
            .descriptor_type(r.resource_type.clone().into())
            .descriptor_count(r.array_size)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
    }
}

impl Drop for PipelineBindings {
    fn drop(&mut self) {
        let vk = self.device.get();

        unsafe {
            vk.destroy_descriptor_set_layout(self.vk_set_layout, None);
        }
    }
}


// Vulkan compute shader pipeline
pub struct Pipeline {
    device            : Rc<VulkanDevice>,
    shader            : Shader,
    descriptor_layout : Vec<PipelineBindings>,
    pipeline_layout   : vk::PipelineLayout,
    pipeline          : vk::Pipeline,
}

impl Pipeline {
    pub fn new(context : &Context, shader : Shader) -> Result<Rc<Self>, String> {
        let vk = context.device.get();

        let set_count = shader.resources().iter().fold(0,
            |accum, (_, r)| cmp::max(r.binding.set + 1, accum));

        let set_iter = (0..set_count).into_iter().map(|i| {
            shader.resources().iter()
                .filter(|(_, r)| r.binding.set == i)
                .map(|(n, r)| (n.clone(), r.clone()))
                .collect::<Vec<_>>()
        });

        let set_layouts = set_iter.map(|bindings| {
            PipelineBindings::new(&context.device, bindings)
        }).collect::<Result<Vec<_>, _>>()?;

        let vk_pipeline_layout = Self::create_pipeline_layout(
                vk, &set_layouts, shader.push_constants())?;

        let vk_pipeline = Self::create_pipeline(vk, shader.code(), vk_pipeline_layout).map_err(|e| {
            unsafe { vk.destroy_pipeline_layout(vk_pipeline_layout, None); }
            e
        })?;

        Ok(Rc::new(Self {
            device            : context.device.clone(),
            shader            : shader,
            descriptor_layout : set_layouts,
            pipeline_layout   : vk_pipeline_layout,
            pipeline          : vk_pipeline,
        }))
    }

    pub fn get_shader<'a>(&'a self) -> &'a Shader {
        &self.shader
    }
}

impl Pipeline {
    fn create_pipeline_layout(vk : &ash::Device, bindings : &[PipelineBindings], push_constants : Option<&StructType>) -> Result<vk::PipelineLayout, String> {
        let set_layouts = bindings.iter().map(|m| m.vk_set_layout).collect::<Vec<_>>();

        let push_data_size = push_constants.map(|p| p.size()).unwrap_or(0);
        
        let push_constant_range = [
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_data_size as u32)
        ];

        let mut create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts);

        if push_constant_range[0].size > 0 {
            create_info = create_info
                .push_constant_ranges(&push_constant_range);
        }

        let vk_pipeline_layout = unsafe {
            vk.create_pipeline_layout(&create_info, None).map_err(
                |e| format!("Failed to create pipeline layout: {}", e.to_string()))?
        };

        Ok(vk_pipeline_layout)
    }

    fn create_pipeline(vk : &ash::Device, code : &[u32], layout : vk::PipelineLayout) -> Result<vk::Pipeline, String> {
        let mut module_info = vk::ShaderModuleCreateInfo::default()
            .code(&code);

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .push_next(&mut module_info)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(c"main");

        let info = [
            vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(layout)
                .base_pipeline_index(-1)
        ];

        let vk_pipelines = unsafe {
            vk.create_compute_pipelines(vk::PipelineCache::null(), &info, None).map_err(
                |(_, e)| format!("Failed to create compute pipeline: {}", e.to_string()))?
        };

        Ok(vk_pipelines[0])
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let vk = self.device.get();

        unsafe {
            vk.destroy_pipeline(self.pipeline, None);
            vk.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

impl Trackable for Pipeline { }

// Vulkan buffer parameters
#[derive(Debug, Default, Clone)]
pub struct BufferInfo {
    pub size        : usize,
    pub cpu_access  : CpuAccess,
}

impl BufferInfo {
    pub fn size(mut self, size : usize) -> Self {
        self.size = size;
        self
    }

    pub fn cpu_access(mut self, access : CpuAccess) -> Self {
        self.cpu_access = access;
        self
    }
}


// Guard around mapped memory region. Will flush mapped range once dropped.
pub struct BufferRegion<'a> {
    buffer : &'a Buffer,
}

impl<'a> BufferRegion<'a> {
    fn new(buffer : &'a Buffer) -> Option<Self> {
        if !buffer.info.cpu_access.contains(CpuAccess::WRITE) {
            return None;
        }

        buffer.invalidate_mapped_range(0, buffer.info.size);
        Some(Self { buffer : buffer })
    }
}

impl Deref for BufferRegion<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        unsafe {
            let ptr = self.buffer.map_ptr.cast::<u8>();
            slice::from_raw_parts(ptr, self.buffer.info.size)
        }
    }
}

impl DerefMut for BufferRegion<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe {
            let ptr = self.buffer.map_ptr.cast::<u8>();
            slice::from_raw_parts_mut(ptr, self.buffer.info.size)
        }
    }
}

impl Drop for BufferRegion<'_> {
    fn drop(&mut self) {
        self.buffer.flush_mapped_range(0, self.buffer.info.size);
    }
}


// Vulkan buffer
pub struct Buffer {
    device      : Rc<VulkanDevice>,
    info        : BufferInfo,
    vk_memory   : vk::DeviceMemory,
    vk_buffer   : vk::Buffer,
    memory_type : u32,
    gpu_address : u64,
    map_ptr     : *mut c_void,
}

impl Buffer {
    pub fn new(context : &Context, info : BufferInfo) -> Result<Rc<Self>, String> {
        let vk = context.device.get();

        let buffer_info = vk::BufferCreateInfo::default()
            .size(info.size as vk::DeviceSize)
            .usage(
                vk::BufferUsageFlags::TRANSFER_DST |
                vk::BufferUsageFlags::TRANSFER_SRC |
                vk::BufferUsageFlags::UNIFORM_BUFFER |
                vk::BufferUsageFlags::STORAGE_BUFFER |
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vk_buffer = unsafe {
            vk.create_buffer(&buffer_info, None).map_err(
                |e| format!("Failed to create buffer: {}", e.to_string()))?
        };

        // Allocate memory on suitable memory type, don't bother suballocating
        let memory_requirements = unsafe {
            vk.get_buffer_memory_requirements(vk_buffer)
        };

        let memory_type = context.device.find_memory_type(memory_requirements.memory_type_bits, info.cpu_access)
            .ok_or("Failed to find memory type for buffer".to_string())?;

        let mut memory_flags = vk::MemoryAllocateFlagsInfo::default()
            .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        let memory_info = vk::MemoryAllocateInfo::default()
            .push_next(&mut memory_flags)
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type);

        let vk_memory = unsafe {
            vk.allocate_memory(&memory_info, None).and_then(|memory| {
                vk.bind_buffer_memory(vk_buffer, memory, 0).map(|_| memory)
            }).map_err(|e| {
                vk.destroy_buffer(vk_buffer, None);
                format!("Failed to allocate buffer memory: {}", e.to_string())
            })?
        };

        // Map memory if requested, otherwise keep null pointer.
        let mut map_ptr : *mut c_void = ptr::null_mut();

        if !info.cpu_access.is_empty() {
            map_ptr = unsafe {
                vk.map_memory(vk_memory, 0,memory_requirements.size, vk::MemoryMapFlags::empty()).map_err(|e| {
                    vk.destroy_buffer(vk_buffer, None);
                    vk.free_memory(vk_memory, None);
                    format!("Failed to map buffer memory: {}", e.to_string())
                })?
            };
        }

        let va_info = vk::BufferDeviceAddressInfo::default()
            .buffer(vk_buffer);

        let gpu_va = unsafe {
            vk.get_buffer_device_address(&va_info)
        };

        Ok(Rc::new(Self {
            device      : context.device.clone(),
            info        : info,
            vk_memory   : vk_memory,
            vk_buffer   : vk_buffer,
            memory_type : memory_type,
            gpu_address : gpu_va,
            map_ptr     : map_ptr,
        }))
    }

    // Buffer properties
    pub fn info<'a>(&'a self) -> &'a BufferInfo {
        &self.info
    }

    // Retrieves mapped slice
    pub fn get_mapped<'a>(&'a self) -> Option<&'a [u8]> {
        if !self.info.cpu_access.contains(CpuAccess::READ) {
            return None;
        }

        self.invalidate_mapped_range(0, self.info.size);

        unsafe {
            let ptr = self.map_ptr.cast::<u8>();
            Some(slice::from_raw_parts(ptr, self.info.size))
        }
    }

    // Retrieves mutable mapped slice
    pub fn get_mapped_mut<'a>(&'a self) -> Option<BufferRegion<'a>> {
        BufferRegion::new(self)
    }

    // Writes raw bytes to mapped memory region
    pub fn write_bytes(&self, offset : usize, data : &[u8]) {
        let size = data.len();

        assert!(offset <= self.info.size && size <= self.info.size - offset);
        assert!(self.info.cpu_access.contains(CpuAccess::WRITE));

        unsafe {
            // SAFETY: Memory is mapped and writable, and we checked the range.
            let ptr = self.map_ptr.byte_add(offset).cast::<u8>();
            ptr::copy(data.as_ptr(), ptr, size);
        }

        self.flush_mapped_range(offset, size);
    }

    // Reads raw bytes from mapped memory region
    pub fn read_bytes(&self, offset : usize, data : &mut [u8]) {
        let size = data.len();

        assert!(offset <= self.info.size && size <= self.info.size - offset);
        assert!(self.info.cpu_access.contains(CpuAccess::READ));

        self.invalidate_mapped_range(offset, size);

        unsafe {
            // SAFETY: Memory is mapped and readable, and we checked the range.
            let ptr = self.map_ptr.byte_add(offset).cast::<u8>();
            ptr::copy(ptr, data.as_mut_ptr(), size);
        }
    }
}

impl Buffer {
    fn flush_mapped_range(&self, offset : usize, size : usize) {
        let memory_type = &self.device.memory.memory_types[self.memory_type as usize];

        if !memory_type.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            let vk = self.device.get();

            let range = [
                vk::MappedMemoryRange::default()
                    .memory(self.vk_memory)
                    .offset(offset as vk::DeviceSize)
                    .size(size as vk::DeviceSize)
            ];

            unsafe {
                if let Err(e) = vk.flush_mapped_memory_ranges(&range) {
                    eprintln!("Failed to flush memory range: {}", e.to_string());
                }
            }
        }
    }

    fn invalidate_mapped_range(&self, offset : usize, size : usize) {
        let memory_type = &self.device.memory.memory_types[self.memory_type as usize];

        if !memory_type.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            let vk = self.device.get();

            let range = [
                vk::MappedMemoryRange::default()
                    .memory(self.vk_memory)
                    .offset(offset as vk::DeviceSize)
                    .size(size as vk::DeviceSize)
            ];

            unsafe {
                if let Err(e) = vk.invalidate_mapped_memory_ranges(&range) {
                    eprintln!("Failed to invalidate memory range: {}", e.to_string());
                }
            }
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let vk = self.device.get();
        
        unsafe {
            vk.destroy_buffer(self.vk_buffer, None);
            vk.free_memory(self.vk_memory, None);
        }
    }
}

impl Trackable for Buffer { }


// Vulkan image parameters
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub dim           : ImageDim,
    pub extent        : (u32, u32, u32),
    pub mips          : u32,
    pub layers        : u32,
    pub format        : ImageFormat,
    pub view_formats  : Vec<ImageFormat>,
}

impl ImageInfo {
    pub fn dim_2d(self, extent : (u32, u32)) -> Self {
        self.dim(ImageDim::Dim2D).extent((extent.0, extent.1, 1))
    }

    pub fn dim(mut self, dim : ImageDim) -> Self {
        self.dim = dim;
        self
    }

    pub fn extent(mut self, extent : (u32, u32, u32)) -> Self {
        self.extent = extent;
        self
    }

    pub fn mips(mut self, mips : u32) -> Self {
        self.mips = mips;
        self
    }

    pub fn layers(mut self, layers : u32) -> Self {
        self.layers = layers;
        self
    }

    pub fn format(mut self, format : ImageFormat) -> Self {
        self.format = format;
        self
    }
}

impl Default for ImageInfo {
    fn default() -> Self {
        Self {
            dim           : ImageDim::Dim2D,
            extent        : (0, 0, 0),
            mips          : 1,
            layers        : 1,
            format        : ImageFormat::RGBA8un,
            view_formats  : vec![],
        }
    }
}


// View parameters
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ImageViewInfo {
    dim               : ImageDim,
    format            : ImageFormat,
    mip_index         : u32,
    mip_count         : u32,
    layer_index       : u32,
    layer_count       : u32,
    usage             : vk::ImageUsageFlags,
}

impl ImageViewInfo {
    fn from_image(image : &Image, resource_type : &ResourceType) -> Result<Self, String> {
        let dim = match resource_type {
            ResourceType::StorageImage(dim, _) |
            ResourceType::SampledImage(dim) => *dim,
            _ => { return Err(format!("Invalid view type for image view: {:?}", resource_type)) }
        };

        let vk_usage = match resource_type {
            ResourceType::StorageImage(_, _) => vk::ImageUsageFlags::STORAGE,
            _ => vk::ImageUsageFlags::SAMPLED,
        };

        let mip_count = match resource_type {
            ResourceType::StorageImage(_, _) => 1,
            _ => image.info.mips
        };

        let layer_count = match dim {
            ImageDim::Dim1D |
            ImageDim::Dim2D |
            ImageDim::Dim2DMS |
            ImageDim::Dim3D |
            ImageDim::DimCube => 1u32,

            ImageDim::Dim1DArray |
            ImageDim::Dim2DArray |
            ImageDim::Dim2DMSArray |
            ImageDim::DimCubeArray => image.info.layers,
        };

        Ok(Self {
            dim     : dim,
            format  : image.info.format,
            mip_index   : 0,
            mip_count   : mip_count,
            layer_index : 0,
            layer_count : layer_count,
            usage       : vk_usage,
        })
    }
}

struct ImageView {
    device            : Rc<VulkanDevice>,
    view              : vk::ImageView,
}

impl ImageView {
    fn new(device : &Rc<VulkanDevice>, image : &Image, info : &ImageViewInfo) -> Result<ImageView, String> {
        let subresources = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(info.layer_index)
            .layer_count(info.layer_count)
            .base_mip_level(info.mip_index)
            .level_count(info.mip_count);

        let create_info = vk::ImageViewCreateInfo::default()
            .image(image.vk_image)
            .view_type(info.dim.into())
            .format(image.info.format.into())
            .subresource_range(subresources);

        let vk = device.get();

        let vk_view = unsafe {
            vk.create_image_view(&create_info, None).map_err(
                |e| format!("Failed to create image view: {}", e.to_string()))?
        };

        let view = Self {
            device      : device.clone(),
            view        : vk_view,
        };

        Ok(view)
    }

    fn get(&self) -> vk::ImageView {
        self.view
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        let vk = self.device.get();

        unsafe {
            vk.destroy_image_view(self.view, None);
        }
    }
}

// Vulkan image
pub struct Image {
    device            : Rc<VulkanDevice>,
    info              : ImageInfo,
    vk_image          : vk::Image,
    vk_memory         : vk::DeviceMemory,
    views             : RefCell<HashMap<ImageViewInfo, ImageView>>,
}

impl Image {
    pub fn new(context : &Context, info : ImageInfo) -> Result<Rc<Self>, String> {
        let vk = context.device.get();

        let format_list : Vec<vk::Format> = info.view_formats
            .iter().map(|&f| f.into()).collect();

        let mut format_list_info = vk::ImageFormatListCreateInfo::default()
            .view_formats(&format_list);

        let (x, y, z) = info.extent;

        let mut image_info = vk::ImageCreateInfo::default()
            .image_type(info.dim.into())
            .format(info.format.into())
            .extent(vk::Extent3D::default()
                .width(x).height(y).depth(z))
            .mip_levels(info.mips)
            .array_layers(info.layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST |
                vk::ImageUsageFlags::TRANSFER_SRC |
                vk::ImageUsageFlags::SAMPLED |
                vk::ImageUsageFlags::STORAGE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        if !format_list.is_empty() {
            image_info = image_info.push_next(&mut format_list_info)
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT);
        }
        
        let vk_image = unsafe {
            vk.create_image(&image_info, None).map_err(
                |e| format!("Failed to create image: {}", e.to_string()))?
        };

        // Allocate memory on suitable memory type, don't bother suballocating
        let memory_requirements = unsafe {
            vk.get_image_memory_requirements(vk_image)
        };

        let memory_type = context.device.find_memory_type(memory_requirements.memory_type_bits, CpuAccess::empty())
            .ok_or("Failed to find memory type for buffer".to_string())?;

        let memory_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type);

        let vk_memory = unsafe {
            vk.allocate_memory(&memory_info, None).and_then(|memory| {
                vk.bind_image_memory(vk_image, memory, 0).map(|_| memory)
            }).map_err(|e| {
                vk.destroy_image(vk_image, None);
                format!("Failed to allocate image memory: {}", e.to_string())
            })?
        };

        Ok(Rc::new(Self {
            device    : context.device.clone(),
            info      : info,
            vk_image  : vk_image,
            vk_memory : vk_memory,
            views     : RefCell::new(HashMap::new()),
        }))
    }

    pub fn create_view<'a>(&'a self, ty : &ResourceType) -> Result<vk::ImageView, String> {
        let mut view_map = self.views.borrow_mut();
        let view_info = ImageViewInfo::from_image(self, ty)?;

        if !view_map.contains_key(&view_info) {
            view_map.insert(view_info.clone(),
                ImageView::new(&self.device, self, &view_info)?);
        }

        Ok(view_map.get(&view_info).unwrap().get())
    }

    // Queries image properties
    pub fn info<'a>(&'a self) -> &'a ImageInfo {
        &self.info
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let vk = self.device.get();
        
        unsafe {
            vk.destroy_image(self.vk_image, None);
            vk.free_memory(self.vk_memory, None);
        }
    }
}

impl Trackable for Image { }


// Sampler parameters
pub struct SamplerInfo {

}


// Vulkan sampler
pub struct Sampler {
    device      : Rc<VulkanDevice>,
    info        : SamplerInfo,
    vk_sampler  : vk::Sampler,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let vk = self.device.get();
        
        unsafe {
            vk.destroy_sampler(self.vk_sampler, None);
        }
    }
}

impl Trackable for Sampler { }


// Thin state tracking
pub struct CommandListState {
    is_begun    : bool,

    sync_stages : vk::PipelineStageFlags2,
    sync_access : vk::AccessFlags2,
}

impl CommandListState {
    fn new() -> Self {
        Self {
            is_begun    : false,
            sync_stages : vk::PipelineStageFlags2::empty(),
            sync_access : vk::AccessFlags2::empty(),
        }
    }
}


// Vulkan objects tied to a queue submission. Provides a command pool,
// command buffer and descriptor pool that can all be recycled, and
// tracks the lifetime of any Vulkan object used while rendering.
pub struct CommandList {
    device              : Rc<VulkanDevice>,

    vk_command_pool     : vk::CommandPool,
    vk_command_buffer   : vk::CommandBuffer,

    vk_descriptor_pool  : vk::DescriptorPool,

    lifetimes           : Vec<Rc<dyn Trackable>>,
    
    state               : CommandListState,

    timeline            : u64,
}

impl CommandList {
    fn new(device : &Rc<VulkanDevice>) -> Result<Self, String> {
        let vk = device.get();

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family);

        let vk_command_pool = unsafe {
            vk.create_command_pool(&command_pool_info, None).map_err(
                |e| format!("Failed to create command pool: {}", e.to_string()))?
        };

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(vk_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let vk_command_buffer = unsafe {
            vk.allocate_command_buffers(&command_buffer_info).map_err(|e| {
                vk.destroy_command_pool(vk_command_pool, None);
                format!("Failed to allocate command buffer: {}", e.to_string())
            })?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER).descriptor_count(1024),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(4096),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(4096),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(4096),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4096),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1024)
            .pool_sizes(&pool_sizes);

        let vk_descriptor_pool = unsafe {
            vk.create_descriptor_pool(&pool_info, None).map_err(|e| {
                vk.destroy_command_pool(vk_command_pool, None);
                format!("Failed to create descriptor pool: {}", e.to_string())
            })?
        };

        Ok(Self {
            device              : device.clone(),
            vk_command_pool     : vk_command_pool,
            vk_command_buffer   : vk_command_buffer[0],
            vk_descriptor_pool  : vk_descriptor_pool,
            lifetimes           : vec![],
            state               : CommandListState::new(),
            timeline            : 0,
        })
    }

    // Adds a resource to track the lifetime of  
    fn track<T : Trackable + 'static>(&mut self, object : &Rc<T>) {
        self.lifetimes.push(object.clone());
    }

    // Resets command buffer, pool etc
    fn reset(&mut self, timeline : &Timeline) -> Result<(), String> {
        timeline.wait(self.timeline)?;

        let vk = self.device.get();
        self.lifetimes.clear();

        unsafe {
            // SAFETY: GPU execution must be tracked externally.
            vk.reset_command_pool(self.vk_command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|e| format!("Failed to reset command pool: {}", e.to_string()))?;
            
            vk.reset_descriptor_pool(self.vk_descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| format!("Failed to reset descriptor pool: {}", e.to_string()))?;
        }

        Ok(())
    }

    // Initializes output image
    fn init_image(&mut self, image : vk::Image, subresources : &vk::ImageSubresourceRange) -> Result<(), String> {
        let cmd = self.begin()?;

        let stage_mask = 
            vk::PipelineStageFlags2::COPY |
            vk::PipelineStageFlags2::COMPUTE_SHADER;

        let access_mask =
            vk::AccessFlags2::TRANSFER_READ |
            vk::AccessFlags2::TRANSFER_WRITE |
            vk::AccessFlags2::SHADER_SAMPLED_READ |
            vk::AccessFlags2::SHADER_STORAGE_READ |
            vk::AccessFlags2::SHADER_STORAGE_WRITE;

        self.emit_barrier(stage_mask, access_mask)?;

        let vk = self.device.get();

        let image_barrier = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(stage_mask)
                .dst_stage_mask(stage_mask)
                .dst_access_mask(access_mask)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(subresources.clone())
        ];

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barrier);

        unsafe {
            vk.cmd_pipeline_barrier2(cmd, &dependency_info);
        }

        Ok(())
    }

    // Records buffer copy command
    fn copy_buffer(&mut self, dst_buffer : vk::Buffer, src_buffer : vk::Buffer, regions : &[vk::BufferCopy]) -> Result<(), String> {
        let cmd = self.begin()?;

        self.emit_barrier(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ |
            vk::AccessFlags2::TRANSFER_WRITE)?;

        let vk = self.device.get();

        unsafe {
            vk.cmd_copy_buffer(cmd, src_buffer, dst_buffer, regions);
        }

        Ok(())
    }

    // Records buffer-to-image copy command
    fn copy_buffer_to_image(&mut self, dst_image : vk::Image, src_buffer : vk::Buffer, regions : &[vk::BufferImageCopy]) -> Result<(), String> {
        let cmd = self.begin()?;

        self.emit_barrier(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ |
            vk::AccessFlags2::TRANSFER_WRITE)?;

        let vk = self.device.get();

        for region in regions {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(region.image_subresource.aspect_mask)
                .base_array_layer(region.image_subresource.base_array_layer)
                .layer_count(region.image_subresource.layer_count)
                .base_mip_level(region.image_subresource.mip_level)
                .level_count(1);

            let image_barrier = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(dst_image)
                    .subresource_range(subresource_range)
            ];

            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(&image_barrier);

            unsafe {
                vk.cmd_pipeline_barrier2(cmd, &dependency_info);
            }
        }

        unsafe {
            vk.cmd_copy_buffer_to_image(cmd, src_buffer, dst_image, vk::ImageLayout::GENERAL, regions);
        }

        Ok(())
    }

    // Records image-to-buffer copy command
    fn copy_image_to_buffer(&mut self, dst_buffer : vk::Buffer, src_image : vk::Image, regions : &[vk::BufferImageCopy]) -> Result<(), String> {
        let cmd = self.begin()?;

        self.emit_barrier(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ |
            vk::AccessFlags2::TRANSFER_WRITE)?;

        let vk = self.device.get();

        unsafe {
            vk.cmd_copy_image_to_buffer(cmd, src_image, vk::ImageLayout::GENERAL, dst_buffer, regions);
        }

        Ok(())
    }

    // Binds a compute pipeline
    fn bind_pipeline(&mut self, pipeline : &Pipeline) -> Result<(), String> {
        let cmd = self.begin()?;
        let vk = self.device.get();

        unsafe {
            vk.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        }

        Ok(())
    }

    // Binds a descriptor set
    fn bind_descriptors(&mut self, pipeline : &Pipeline, index : u32, set : vk::DescriptorSet) -> Result<(), String> {
        let cmd = self.begin()?;
        let vk = self.device.get();

        let sets = [set];

        unsafe {
            vk.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout, index, &sets, &[]);
        }

        Ok(())
    }

    // Sets push data for the given compute pipeline
    fn set_push_data(&mut self, pipeline : &Pipeline, data : &[u8]) -> Result<(), String> {
        let push_data_size = pipeline.shader.push_constants()
            .map(|p| p.size()).unwrap_or(0);

        if push_data_size == 0 {
            return Ok(());
        }

        let cmd = self.begin()?;

        let vk = self.device.get();
        let vk_layout = pipeline.pipeline_layout;

        unsafe {
            vk.cmd_push_constants(cmd, vk_layout,
                vk::ShaderStageFlags::COMPUTE, 0, &data[0..push_data_size]);
        }

        Ok(())
    }

    // Dispatches bound compute pipeline
    fn dispatch(&mut self, workgroups : (u32, u32, u32)) -> Result<(), String> {
        self.emit_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::UNIFORM_READ |
            vk::AccessFlags2::SHADER_SAMPLED_READ |
            vk::AccessFlags2::SHADER_STORAGE_READ |
            vk::AccessFlags2::SHADER_STORAGE_WRITE)?;

        let cmd = self.begin()?;
        let vk = self.device.get();

        unsafe {
            let (x, y, z) = workgroups;
            vk.cmd_dispatch(cmd, x, y, z);
        }

        Ok(())
    }

    // Submits the command buffer to the queue using the
    // given timeline for synchronization purposes.
    fn submit<'a>(&mut self, timeline : &'a Timeline) -> Result<SyncPoint<'a>, String> {
        let cmd = self.begin()?;

        self.emit_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER |
            vk::PipelineStageFlags2::COPY |
            vk::PipelineStageFlags2::CLEAR |
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::UNIFORM_READ |
            vk::AccessFlags2::SHADER_SAMPLED_READ |
            vk::AccessFlags2::SHADER_STORAGE_READ |
            vk::AccessFlags2::SHADER_STORAGE_WRITE |
            vk::AccessFlags2::TRANSFER_READ |
            vk::AccessFlags2::TRANSFER_WRITE |
            vk::AccessFlags2::HOST_READ)?;

        self.close()?;

        // Update timeline and prepare for queue submission
        self.timeline = timeline.vk_timeline;

        let command_buffer_info = [
            vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd)
        ];

        let semaphore_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(timeline.vk_semaphore)
                .value(timeline.vk_timeline)
        ];

        let submit_info = [
            vk::SubmitInfo2::default()
                .command_buffer_infos(&command_buffer_info)
                .signal_semaphore_infos(&semaphore_info)
        ];

        let vk = self.device.get();

        unsafe {
            vk.queue_submit2(self.device.vk_queue, &submit_info, vk::Fence::null()).map_err(
                |e| format!("Failed to submit command list: {}", e.to_string()))?;
        }

        Ok(SyncPoint::new(timeline, timeline.vk_timeline))
    }

    // Closes current command buffer.
    fn close(&mut self) -> Result<(), String> {
        let vk = self.device.get();

        if self.state.is_begun {
            unsafe {
                vk.end_command_buffer(self.vk_command_buffer).map_err(
                    |e| format!("Failed to end command buffer: {}", e.to_string()))?;
            }

            self.state = CommandListState::new();
        }

        Ok(())
    }

    // Retrieves and begins the command buffer
    fn begin(&mut self) -> Result<vk::CommandBuffer, String> {
        if !self.state.is_begun {
            let vk = self.device.get();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe {
                vk.begin_command_buffer(self.vk_command_buffer, &begin_info).map_err(
                    |e| format!("Failed to begin command buffer: {}", e.to_string()))?;
            }

            self.state.is_begun = true;
        }

        Ok(self.vk_command_buffer)
    }

    // Submits barrier for the next command. All commands are serialized
    // for simplicity.
    fn emit_barrier(&mut self, stage_mask : vk::PipelineStageFlags2, access_mask : vk::AccessFlags2) -> Result<(), String> {
        let cmd = self.begin()?;

        if !self.state.sync_stages.is_empty() {
            let barrier = [
                vk::MemoryBarrier2::default()
                    .src_stage_mask(self.state.sync_stages)
                    .src_access_mask(self.state.sync_access)
                    .dst_stage_mask(stage_mask)
                    .dst_access_mask(access_mask)
            ];

            let dependency = vk::DependencyInfo::default()
                .memory_barriers(&barrier);

            unsafe {
                let vk = self.device.get();
                vk.cmd_pipeline_barrier2(cmd, &dependency);
            }
        }

        self.state.sync_stages = stage_mask;
        self.state.sync_access = access_mask;

        Ok(())
    }

    // Allocates a descriptor set.
    fn allocate_set(&self, layout : &PipelineBindings) -> Result<vk::DescriptorSet, String> {
        let vk = self.device.get();

        let set_layout = [layout.vk_set_layout];

        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.vk_descriptor_pool)
            .set_layouts(&set_layout);

        let set = unsafe {
            vk.allocate_descriptor_sets(&set_info).map_err(
                |e| format!("Failed to allocate descriptor set: {}", e.to_string()))?
        };

        Ok(set[0])
    }

    // Updates a descriptor set
    fn write_descriptors(&self, writes : &[vk::WriteDescriptorSet]) {
        let vk = self.device.get();

        unsafe {
            vk.update_descriptor_sets(&writes, &[]);
        }
    }
}

impl Drop for CommandList {
    fn drop(&mut self) {
        let vk = self.device.get();

        unsafe {
            vk.destroy_descriptor_pool(self.vk_descriptor_pool, None);
            vk.destroy_command_pool(self.vk_command_pool, None);
        }
    }
}


// Timeline semaphore for synchronization
struct Timeline {
    device        : Rc<VulkanDevice>,
    vk_semaphore  : vk::Semaphore,
    vk_timeline   : u64,
}

impl Timeline {
    fn new(device : &Rc<VulkanDevice>) -> Result<Self, String> {
        let vk = device.get();

        let mut semaphore_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE);

        let semaphore_info = vk::SemaphoreCreateInfo::default()
            .push_next(&mut semaphore_type_info);

        let vk_semaphore = unsafe {
            vk.create_semaphore(&semaphore_info, None).map_err(
                |e| format!("Failed to create semaphore: {}", e.to_string()))?
        };

        Ok(Self {
            device        : device.clone(),
            vk_semaphore  : vk_semaphore,
            vk_timeline   : 0,
        })
    }

    fn wait(&self, timeline : u64) -> Result<(), String> {
        let vk = self.device.get();

        let sem = [self.vk_semaphore];
        let val = [timeline];

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sem)
            .values(&val);

        unsafe {
            vk.wait_semaphores(&wait_info, u64::MAX).map_err(
                |e| format!("Failed to create semaphore: {}", e.to_string()))?;
        }

        Ok(())
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        let vk = self.device.get();

        unsafe {
            // SAFETY: Caller must be sure to wait for semaphore
            // be signaled to max pending value first.
            vk.destroy_semaphore(self.vk_semaphore, None);
        }
    }
}


// Sync point as a convenience API to wait for the global timeline.
pub struct SyncPoint<'a> {
    timeline    : &'a Timeline,
    value       : u64,
}

impl SyncPoint<'_> {
    pub fn wait(&self) -> Result<(), String> {
        self.timeline.wait(self.value)
    }
}

impl<'a> SyncPoint<'a> {
    fn new(timeline : &'a Timeline, value : u64) -> Self {
        Self {
            timeline  : timeline,
            value     : value,
        }
    }
}


// Vulkan context
pub struct Context {
    device      : Rc<VulkanDevice>,
    timeline    : Timeline,

    lists       : Vec<CommandList>,
    list_index  : usize,
}

impl Context {
    pub fn new(info : VulkanInfo) -> Result<Self, String> {
        let instance = Rc::new(VulkanInstance::new(&info)?);
        let device = Rc::new(VulkanDevice::new(&instance, &info)?);
        let timeline = Timeline::new(&device)?;

        let command_lists = (0..4).into_iter().map(|_| {
            CommandList::new(&device)
        }).collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            device      : device,
            timeline    : timeline,
            lists       : command_lists,
            list_index  : 0,
        })
    }

    // Submit pending commands. Returns a sync point to wait for all
    // submitted commands to complete, which can be useful for resource
    // read-back.
    pub fn submit<'a>(&'a mut self) -> Result<SyncPoint<'a>, String> {
        // Advance submission counter
        self.timeline.vk_timeline += 1;
        
        // Submit current command list
        let list = &mut self.lists[self.list_index];
        let sync_point = list.submit(&self.timeline)?;

        // Advance to next list and reset it
        self.list_index = (self.list_index + 1) % self.lists.len();

        let list = &mut self.lists[self.list_index];
        list.reset(&self.timeline)?;

        Ok(sync_point)
    }

    // Initializes image
    pub fn init_image(&mut self,
        dst_image : &Rc<Image>
    ) -> Result<(), String> {
        let list = self.get_list();

        list.track(dst_image);

        let subresources = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(dst_image.info().layers)
            .base_mip_level(0)
            .level_count(dst_image.info().mips);

        list.init_image(dst_image.vk_image, &subresources)
    }

    // Copies buffer data
    pub fn copy_buffer(&mut self,
        dst_buffer : &Rc<Buffer>,
        src_buffer : &Rc<Buffer>
    ) -> Result<(), String> {
        if dst_buffer.info().size != src_buffer.info().size {
            return Err(format!("Cannot copy buffer of size {} to buffer of size {}",
                src_buffer.info().size, dst_buffer.info().size));
        }

        let buffer_region = [
            vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(dst_buffer.info().size as vk::DeviceSize)
        ];

        let list = self.get_list();

        list.track(dst_buffer);
        list.track(src_buffer);

        list.copy_buffer(
            dst_buffer.vk_buffer,
            src_buffer.vk_buffer,
            &buffer_region)
    }

    // Copies buffer data to an image
    pub fn copy_buffer_to_image(&mut self,
        dst_image   : &Rc<Image>,
        dst_mip     : u32,
        dst_layer   : u32,
        src_buffer  : &Rc<Buffer>
    ) -> Result<(), String> {
        let vk_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(dst_mip)
            .base_array_layer(dst_layer)
            .layer_count(1);

        let (w, h, d) = dst_image.info().extent;

        let vk_offset = vk::Offset3D::default();
        let vk_extent = vk::Extent3D::default()
            .width(cmp::max(1, w >> dst_mip))
            .height(cmp::max(1, h >> dst_mip))
            .depth(cmp::max(1, d >> dst_mip));

        let copy_region = [
            vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk_subresource)
                .image_offset(vk_offset)
                .image_extent(vk_extent)
        ];

        let list = self.get_list();

        list.track(dst_image);
        list.track(src_buffer);

        list.copy_buffer_to_image(
            dst_image.vk_image,
            src_buffer.vk_buffer,
            &copy_region)
    }

    // Copies image data to a buffer
    pub fn copy_image_to_buffer(&mut self,
        dst_buffer  : &Rc<Buffer>,
        src_image   : &Rc<Image>,
        src_mip     : u32,
        src_layer   : u32,
    ) -> Result<(), String> {
        let vk_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(src_mip)
            .base_array_layer(src_layer)
            .layer_count(1);

        let (w, h, d) = src_image.info().extent;

        let vk_offset = vk::Offset3D::default();
        let vk_extent = vk::Extent3D::default()
            .width(cmp::max(1, w >> src_mip))
            .height(cmp::max(1, h >> src_mip))
            .depth(cmp::max(1, d >> src_mip));

        let copy_region = [
            vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk_subresource)
                .image_offset(vk_offset)
                .image_extent(vk_extent)
        ];

        let list = self.get_list();

        list.track(dst_buffer);
        list.track(src_image);

        list.copy_image_to_buffer(
            dst_buffer.vk_buffer,
            src_image.vk_image,
            &copy_region)
    }

    // Dispatches compute pipeline with the given resources and shader arguments.
    pub fn dispatch(&mut self,
        pipeline        : &Rc<Pipeline>,
        workgroups      : (u32, u32, u32),
        shader_args     : &sj::Value,
        shader_bindings : &HashMap<String, Binding>,
    ) -> Result<(), String> {
        let list = self.get_list();

        // Track and bind shader
        list.track(pipeline);
        list.bind_pipeline(pipeline)?;

        // Bind and track resources for each descriptor set
        for (index, bindings) in pipeline.descriptor_layout.iter().enumerate() {
            if bindings.resources.is_empty() {
                continue;
            }

            // Allocate descriptor set
            let vk_set = list.allocate_set(bindings)?;

            for (name, binding) in &bindings.resources {
                let Some(resource) = shader_bindings.get(name) else {
                    return Err(format!("Resource '{name}' missing."));
                };

                let mut image_info = [vk::DescriptorImageInfo::default()];
                let mut buffer_info = [vk::DescriptorBufferInfo::default()];

                let descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_set(vk_set)
                    .dst_binding(binding.binding.binding)
                    .dst_array_element(0)
                    .descriptor_count(1)
                    .descriptor_type(binding.resource_type.clone().into());

                let descriptor_write = [
                    match binding.resource_type {
                        ResourceType::Sampler => {
                            match resource {
                                Binding::Sampler(s) => {
                                    list.track(s);

                                    image_info = [
                                        vk::DescriptorImageInfo::default()
                                            .sampler(s.vk_sampler)
                                    ];
                                },
                                _ => {
                                    // Null samplers are explicitly disallowed
                                    return Err(format!("Resource '{name}' expected to be sampler, but got:\n{:#?}", binding.resource_type));
                                }
                            }

                            descriptor_write.image_info(&image_info)
                        },

                        ResourceType::SampledImage(_) |
                        ResourceType::StorageImage(_, _) => {
                            match resource {
                                Binding::Null => { },
                                Binding::Image(v) => {
                                    list.track(v);

                                    image_info = [
                                        vk::DescriptorImageInfo::default()
                                            .image_view(v.create_view(&binding.resource_type)?)
                                            .image_layout(vk::ImageLayout::GENERAL)
                                    ];
                                },
                                _ => {
                                    // Null samplers are explicitly disallowed
                                    return Err(format!("Resource '{name}' expected to be image, but got:\n{:#?}", binding.resource_type));
                                }
                            }

                            descriptor_write.image_info(&image_info)
                        },

                        ResourceType::UniformBuffer(_) |
                        ResourceType::StorageBuffer(_) => {
                            match resource {
                                Binding::Null => { },
                                Binding::Buffer(b) => {
                                    list.track(b);

                                    buffer_info = [
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(b.vk_buffer)
                                            .offset(0)
                                            .range(b.info.size as vk::DeviceSize)
                                    ];
                                },
                                _ => {
                                    return Err(format!("Resource '{name}' expected to be buffer, but got:\n{:#?}", binding.resource_type));
                                }
                            }

                            descriptor_write.buffer_info(&buffer_info)
                        },
                    }
                ];

                list.write_descriptors(&descriptor_write);
            }

            list.bind_descriptors(pipeline, index as u32, vk_set)?;
        }

        // Update shader arguments as push constants
        if let Some(ty) = pipeline.shader.push_constants() {
            let mut arg_data = [0u8; 256];

            from_json_into(&mut arg_data, shader_args,
                &DataType::Struct(ty.clone()))?;

            for m in &ty.members {
                if let DataType::Pointer(_) = m.data_type {
                    let a = m.offset;
                    let b = m.offset + mem::size_of::<u64>();

                    let gpu_va = shader_bindings.get(&m.name)
                        .and_then(|r| match r {
                            // Track buffer resource
                            Binding::Buffer(b) => {
                                list.track(b);
                                Some(b.gpu_address)
                            },
                            _ => None
                        }).ok_or_else(|| {
                            // Passing null VA is likely unexpected by
                            // the shader, so make this a hard error
                            format!("Resource '{}' missing or not a buffer", m.name)
                        })?;

                    arg_data[a..b].copy_from_slice(&gpu_va.to_le_bytes());
                }
            }

            list.set_push_data(pipeline, &arg_data)?;
        }

        list.dispatch(workgroups)
    }
}

impl Context {
    fn get_list<'a>(&'a mut self) -> &'a mut CommandList {
        &mut self.lists[self.list_index]
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // Wait for all pending GPU work to complete first
        if let Err(e) = self.timeline.wait(self.timeline.vk_timeline) {
            eprintln!("Failed to wait for semaphore: {}", e.to_string());
        }
    }
}

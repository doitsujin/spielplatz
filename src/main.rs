use std::env;
use std::path::{Path};

mod batch;
mod data_conv;
mod shader;
mod shader_reflection;
mod vulkan;

use batch::*;
use vulkan::*;

fn run_app() -> Result<(), String> {
    let args : Vec<_> = env::args().collect();

    let json_path = Path::new(args.get(1).ok_or("Missing arg".to_string())?);
    let batch_file = BatchFile::parse_file(&json_path)?;

    let mut vulkan = Context::new(VulkanInfo::new().debug(true))?;

    let base_path = json_path.parent().ok_or("Invalid path".to_string())?;
    let shaders = batch_file.load_shaders(&base_path, &vulkan)?;

    for batch in batch_file.list_batches() {
        println!("{batch}");

        let instance = batch_file.load_batch(&base_path,
            &mut vulkan, &shaders, &batch)?;

        instance.dispatch(&mut vulkan)?;
        instance.store_outputs(&mut vulkan)?;
    }

    Ok(())
}

fn main() {
    if let Err(e) = run_app() {
        eprintln!("{e}");
    }
}

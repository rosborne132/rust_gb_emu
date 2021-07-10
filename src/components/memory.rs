pub struct Memory {
  pub video_ram: Vec<u8>,
  pub work_ram_0: Vec<u8>,
  pub work_ram_1: Vec<u8>,
  pub other_ram: Vec<u8>,
  pub external_ram: Vec<u8>,
  pub external_ram_enabled: bool,
}
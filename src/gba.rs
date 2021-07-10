use std::collections::BTreeMap;
use std::collections::HashSet;
use std::io;
use std::io::Write;
use std::panic;
use std::result::Result;

use super::components::cartridge::ROM;
use super::components::opcode::{flag_picker, FlagBit, MemoryBankType, Opcode, Register};
use super::components::memory::Memory;

// Used for debugging.
// static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

mod constants;
mod math;

#[derive(Copy, Clone, Debug)]
pub struct RegisterState {
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
    pub e: u8,
    pub f: u8,
    pub h: u8,
    pub l: u8,
}

#[derive(Copy, Clone, Debug)]
pub struct ProgramState {
    pub stack_pointer: u16,
    pub program_counter: u16,
    pub cycle_count: u64,
    rom_bank: u8,
}

#[derive(Copy, Clone, Debug)]
pub struct Interrupts {
    pub master_enabled: bool,
    pub request_flag: u8,
    pub enable_flag: u8,
    pub halted: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InterruptBit {
    VBlank,
    LCDStatus,
    Timer,
    Serial,
    Joypad,
}

static INTERRUPTS: [InterruptBit; 5] = [
    InterruptBit::VBlank,
    InterruptBit::LCDStatus,
    InterruptBit::Timer,
    InterruptBit::Serial,
    InterruptBit::Joypad,
];

fn interrupt_picker(bit: InterruptBit) -> u8 {
    match bit {
        InterruptBit::VBlank => 1 << 0,
        InterruptBit::LCDStatus => 1 << 1,
        InterruptBit::Timer => 1 << 2,
        InterruptBit::Serial => 1 << 3,
        InterruptBit::Joypad => 1 << 4,
    }
}

fn interrupt_address(bit: InterruptBit) -> u16 {
    match bit {
        InterruptBit::VBlank => 0x40,
        InterruptBit::LCDStatus => 0x48,
        InterruptBit::Timer => 0x50,
        InterruptBit::Serial => 0x58,
        InterruptBit::Joypad => 0x60,
    }
}

pub enum LCDCInterruptBit {
    LYCLYCoincidence = 1 << 6,
    OAM = 1 << 5,
    VBlank = 1 << 4,
    HBlank = 1 << 3,
}

pub struct ScreenOutput {
    screen: Vec<u8>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum JoypadButton {
    Down,
    Up,
    Left,
    Right,
    A,
    B,
    Start,
    Select,
}

fn joypad_button_type(button: JoypadButton) -> JoypadSelectFilter {
    match button {
        JoypadButton::Down => JoypadSelectFilter::Direction,
        JoypadButton::Up => JoypadSelectFilter::Direction,
        JoypadButton::Left => JoypadSelectFilter::Direction,
        JoypadButton::Right => JoypadSelectFilter::Direction,
        JoypadButton::A => JoypadSelectFilter::Button,
        JoypadButton::B => JoypadSelectFilter::Button,
        JoypadButton::Start => JoypadSelectFilter::Button,
        JoypadButton::Select => JoypadSelectFilter::Button,
    }
}

#[derive(PartialEq, Eq, Hash)]
pub enum JoypadSelectFilter {
    Undetermined,
    Button,
    Direction,
}

pub struct JoypadInput {
    pressed_keys: HashSet<JoypadButton>,
    select: JoypadSelectFilter,
}

pub struct Interpreter {
    pub rom: ROM,
    pub register_state: RegisterState,
    pub program_state: ProgramState,
    memory: Memory,
    pub output: String,
    pub interrupts: Interrupts,
    pub screen_output: ScreenOutput,
    pub joypad_input: JoypadInput,
}

impl Interpreter {
    pub fn with_rom(rom: ROM) -> Interpreter {
        if !rom.has_valid_header_checksum() || !rom.has_nintendo_logo() {
            panic!("invalid header");
        }
        let ram_size = rom.ram_size();
        let mut ret = Interpreter {
            rom,
            register_state: RegisterState {
                a: 0x01,
                b: 0x00,
                c: 0x13,
                d: 0x00,
                e: 0xd8,
                f: 0xb0,
                h: 0x01,
                l: 0x4d,
            },
            program_state: ProgramState {
                stack_pointer: 0xfffe,
                program_counter: 0x100,
                cycle_count: 0,
                rom_bank: 0,
            },
            memory: Memory {
                video_ram: vec![0; 0x2000],
                work_ram_0: vec![0; 0x1000],
                work_ram_1: vec![0; 0x1000],
                other_ram: vec![0; 0x10000],
                external_ram: vec![0; ram_size],
                external_ram_enabled: false,
            },
            output: String::from(""),
            interrupts: Interrupts {
                master_enabled: false,
                request_flag: 0xe0,
                enable_flag: 0x00,
                halted: false,
            },
            screen_output: ScreenOutput {
                screen: vec![0; 160 * 144],
            },
            joypad_input: JoypadInput {
                pressed_keys: HashSet::new(),
                select: JoypadSelectFilter::Undetermined,
            },
        };
        ret.save_memory(constants::LCDC, 0x91);
        ret
    }

    fn opcode(&self, address: u16) -> (Opcode, u16, u16) {
        let reader = |address| self.read_memory(address);
        self.rom.opcode(address, reader)
    }

    pub fn get_next_instructions(&self) -> BTreeMap<u16, Opcode> {
        let mut pc = self.program_state.program_counter;
        let mut result = BTreeMap::new();
        for _i in 0..3 {
            let (opcode, opcode_size, _cycle_count) = self.opcode(pc);
            result.insert(pc, opcode);
            pc += opcode_size;
        }
        return result;
    }

    pub fn run_single_instruction(&mut self) -> () {
        let current_pc = self.program_state.program_counter;
        let (opcode, opcode_size, mut cycle_cost) = self.opcode(current_pc);

        // There are two ways an interrupt can be triggered:
        // 1) The program has enabled interrupts, and run into an action causing an interrupt
        //    that it cares about.
        // 2) The program has halted, and an interrupt is triggered.
        let interrupt_normally = self.interrupts.master_enabled
            && ((0x1f & self.interrupts.enable_flag & self.interrupts.request_flag) != 0);
        let interrupt_halted = self.interrupts.halted && (self.interrupts.request_flag != 0);
        if interrupt_normally || interrupt_halted {
            let old_master_enabled = self.interrupts.master_enabled;
            self.interrupts.master_enabled = false;
            self.interrupts.halted = false;
            for bit in INTERRUPTS.iter() {
                let picker = interrupt_picker(*bit);
                if self.interrupts.enable_flag & self.interrupts.request_flag & picker != 0 {
                    if old_master_enabled {
                        let address = interrupt_address(*bit);
                        self.interrupts.request_flag = !picker & self.interrupts.request_flag;
                        self.program_state.program_counter = self.do_call(address).unwrap();
                    }
                    return;
                }
            }
            // TODO: fix timing
            // It takes 20 clocks to dispatch an interrupt.
            // If CPU is in HALT mode, another extra 4 clocks are needed

            // TODO: implement halt bug
            if !interrupt_halted {
                panic!("could not find value to interrupt to");
            }
        }

        if self.interrupts.halted {
            self.handle_cycle_increment(cycle_cost);
            return;
        }

        // PC should be updated before we actually run the instruction.
        // This matters when you store return pointers on the stack.
        self.program_state.program_counter = current_pc.wrapping_add(opcode_size);

        let mut jump_location: Option<u16> = None;
        match opcode {
            Opcode::Noop => (),
            Opcode::Jump(address) => jump_location = Some(address),
            Opcode::JumpCond(flag, set, address) => {
                cycle_cost = 12;
                if self.get_flag(flag) == set {
                    jump_location = Some(address);
                    cycle_cost = 16;
                }
            }
            Opcode::JumpHL => {
                jump_location = Some(self.register_pair_value(Register::H, Register::L))
            }
            Opcode::JumpRelative(relative_address) => {
                jump_location = Some(
                    ((self.program_state.program_counter as i32) + (relative_address as i32))
                        as u16,
                );
            }
            Opcode::JumpRelativeCond(flag, set, relative_address) => {
                cycle_cost = 8;
                if self.get_flag(flag) == set {
                    jump_location = Some(
                        ((self.program_state.program_counter as i32) + (relative_address as i32))
                            as u16,
                    );
                    cycle_cost = 12;
                }
            }
            Opcode::DisableInterrupts => self.interrupts.master_enabled = false,
            Opcode::EnableInterrupts => self.interrupts.master_enabled = true,
            Opcode::Load8(register, value) => {
                self.handle_save_register(register, value);
            }
            Opcode::Load16(hi_register, lo_register, value) => {
                self.save_register_pair(hi_register, lo_register, value);
            }
            Opcode::LoadAddress(register, address) => {
                let value = self.load_address(address);
                self.handle_save_register(register, value);
            }
            Opcode::SaveRegister(register, address) => {
                let value = self.get_register_value(register);
                self.save_memory(address, value);
            }
            Opcode::LoadRamCIntoA => {
                let address = 0xff00 + (self.get_register_value(Register::C) as u16);
                self.handle_save_register(Register::A, self.read_memory(address));
            }
            Opcode::SaveAIntoRamC => {
                let address = 0xff00 + (self.get_register_value(Register::C) as u16);
                self.save_memory(address, self.get_register_value(Register::A));
            }
            Opcode::Call(address) => jump_location = self.do_call(address),
            Opcode::CallCond(flag, set, address) => {
                cycle_cost = 12;
                if self.get_flag(flag) == set {
                    jump_location = self.do_call(address);
                    cycle_cost = 24;
                }
            }
            Opcode::Restart(address) => jump_location = self.do_call(address),
            Opcode::Return => jump_location = self.do_return(),
            Opcode::ReturnCond(flag, set) => {
                cycle_cost = 8;
                if self.get_flag(flag) == set {
                    jump_location = self.do_return();
                    cycle_cost = 20;
                }
            }
            Opcode::ReturnInterrupt => {
                jump_location = self.do_return();
                self.interrupts.master_enabled = true;
            }
            Opcode::LoadReg(to_register, from_register) => {
                self.handle_save_register(to_register, self.get_register_value(from_register));
            }
            Opcode::LoadAddressFromRegisters(to_register, hi_addr, lo_addr) => {
                let address = self.register_pair_value(hi_addr, lo_addr);
                self.handle_save_register(to_register, self.load_address(address));
            }
            Opcode::LoadRegisterIntoMemory(from_register, hi_addr, lo_addr) => {
                let address = self.register_pair_value(hi_addr, lo_addr);
                self.save_memory(address, self.get_register_value(from_register));
            }
            Opcode::Push(hi_register, lo_register) => {
                let old_sp = self.program_state.stack_pointer;
                self.save_memory(old_sp - 1, self.get_register_value(hi_register));
                self.save_memory(old_sp - 2, self.get_register_value(lo_register));
                self.program_state.stack_pointer -= 2;
            }
            Opcode::Pop(hi_register, lo_register) => {
                let old_sp = self.program_state.stack_pointer;
                self.handle_save_register(lo_register, self.read_memory(old_sp));
                self.handle_save_register(hi_register, self.read_memory(old_sp + 1));
                self.program_state.stack_pointer += 2;
            }
            Opcode::AddHL(hi_register, lo_register) => {
                let a = self.register_pair_value(Register::H, Register::L);
                let b = self.register_pair_value(hi_register, lo_register);
                let result = math::add16(a, b);
                self.save_register_pair(Register::H, Register::L, result.value);
                self.set_flag(FlagBit::AddSub, result.add_sub.unwrap());
                self.set_flag(FlagBit::Carry, result.carry.unwrap());
                self.set_flag(FlagBit::HalfCarry, result.half_carry.unwrap());
            }
            Opcode::AddSP(value) => {
                let result = math::sp_add(self.program_state.stack_pointer, value);
                self.program_state.stack_pointer = result.value;
                self.apply_math_result16_flags(result);
            }
            Opcode::SaveHLSP(value) => {
                let result = math::sp_add(self.program_state.stack_pointer, value);
                self.apply_math_result16_flags(result);
                self.save_register_pair(Register::H, Register::L, result.value);
            }
            Opcode::IncPair(hi_register, lo_register) => {
                let value = self
                    .register_pair_value(hi_register, lo_register)
                    .wrapping_add(1);
                self.save_register_pair(hi_register, lo_register, value);
            }
            Opcode::Inc(register) => {
                let old_value = self.get_register_value(register);
                let new_value = old_value.wrapping_add(1);
                self.set_flag(FlagBit::Zero, new_value == 0);
                self.set_flag(FlagBit::AddSub, false);
                self.set_half_carry_add(old_value, 1);
                self.handle_save_register(register, new_value);
            }
            Opcode::DecPair(hi_register, lo_register) => {
                let value = self
                    .register_pair_value(hi_register, lo_register)
                    .wrapping_sub(1);
                self.save_register_pair(hi_register, lo_register, value);
            }
            Opcode::Dec(register) => {
                let old_value = self.get_register_value(register);
                let new_value = old_value.wrapping_sub(1);
                self.set_flag(FlagBit::Zero, new_value == 0);
                self.set_flag(FlagBit::AddSub, true);
                self.set_half_carry_sub(old_value, 1);
                self.handle_save_register(register, new_value);
            }
            Opcode::LoadHLInc => {
                let address = self.register_pair_value(Register::H, Register::L);
                self.handle_save_register(Register::A, self.load_address(address));
                self.save_register_pair(Register::H, Register::L, address + 1);
            }
            Opcode::SaveHLInc => {
                let address = self.register_pair_value(Register::H, Register::L);
                self.save_memory(address, self.get_register_value(Register::A));
                self.save_register_pair(Register::H, Register::L, address + 1);
            }
            Opcode::SaveSP(address) => {
                let lower = (self.program_state.stack_pointer & 0xff) as u8;
                let upper = ((self.program_state.stack_pointer & 0xff00) >> 8) as u8;
                self.save_memory(address, lower);
                self.save_memory(address + 1, upper);
            }
            Opcode::LoadHLDec => {
                let address = self.register_pair_value(Register::H, Register::L);
                self.handle_save_register(Register::A, self.load_address(address));
                self.save_register_pair(Register::H, Register::L, address - 1);
            }
            Opcode::SaveHLDec => {
                let address = self.register_pair_value(Register::H, Register::L);
                self.save_memory(address, self.get_register_value(Register::A));
                self.save_register_pair(Register::H, Register::L, address - 1);
            }
            Opcode::LoadHLIntoSP => {
                let address = self.register_pair_value(Register::H, Register::L);
                self.program_state.stack_pointer = address;
            }
            Opcode::DAA => {
                let result = math::daa(
                    self.get_register_value(Register::A),
                    self.get_flag(FlagBit::Carry),
                    self.get_flag(FlagBit::HalfCarry),
                    self.get_flag(FlagBit::AddSub),
                );
                self.handle_save_register(Register::A, result.value);
                self.apply_math_result_flags(result);
            }
            Opcode::And(register) => self.do_math_reg(register, math::and),
            Opcode::Or(register) => self.do_math_reg(register, math::or),
            Opcode::Xor(register) => self.do_math_reg(register, math::xor),
            Opcode::AndValue(value) => self.do_math(value, math::and),
            Opcode::OrValue(value) => self.do_math(value, math::or),
            Opcode::XorValue(value) => self.do_math(value, math::xor),
            Opcode::AddValue(value) => self.do_math(value, math::add),
            Opcode::AddCarryValue(value) => self.do_math_carry(value, math::adc),
            Opcode::SubCarryValue(value) => self.do_math_carry(value, math::sbc),
            Opcode::SubValue(value) => self.do_math(value, math::sub),
            Opcode::CpValue(value) => self.do_math(value, math::cp),
            Opcode::Cp(register) => self.do_math_reg(register, math::cp),
            Opcode::Add(register) => self.do_math_reg(register, math::add),
            Opcode::AddCarry(register) => self.do_math_carry_reg(register, math::adc),
            Opcode::SubCarry(register) => self.do_math_carry_reg(register, math::sbc),
            Opcode::Sub(register) => self.do_math_reg(register, math::sub),
            Opcode::RLC(register) => self.do_bit_op(register, math::rlc),
            Opcode::RRC(register) => self.do_bit_op(register, math::rrc),
            Opcode::RR(register) => self.do_bit_op_carry(register, math::rr),
            Opcode::RL(register) => self.do_bit_op_carry(register, math::rl),
            Opcode::RLCA => {
                self.do_bit_op(Register::A, math::rlc);
                self.set_flag(FlagBit::Zero, false);
            }
            Opcode::RRCA => {
                self.do_bit_op(Register::A, math::rrc);
                self.set_flag(FlagBit::Zero, false);
            }
            Opcode::RRA => {
                self.do_bit_op_carry(Register::A, math::rr);
                self.set_flag(FlagBit::Zero, false);
            }
            Opcode::RLA => {
                self.do_bit_op_carry(Register::A, math::rl);
                self.set_flag(FlagBit::Zero, false);
            }
            Opcode::SLA(register) => self.do_bit_op(register, math::sla),
            Opcode::SRA(register) => self.do_bit_op(register, math::sra),
            Opcode::SRL(register) => self.do_bit_op(register, math::srl),
            Opcode::Swap(register) => self.do_bit_op(register, math::swap),
            Opcode::Bit(register, pos) => {
                let value = self.get_register_value(register);
                let set = (value & (0x1 << pos)) != 0x0;
                self.set_flag(FlagBit::Zero, !set);
                self.set_flag(FlagBit::AddSub, false);
                self.set_flag(FlagBit::HalfCarry, true);
            }
            Opcode::Set(register, pos) => {
                let value = self.get_register_value(register);
                let new_val = value | (0x1 << pos);
                self.handle_save_register(register, new_val);
            }
            Opcode::Reset(register, pos) => {
                let value = self.get_register_value(register);
                let new_val = value & !(0x1 << pos);
                self.handle_save_register(register, new_val);
            }
            Opcode::CPL => {
                let value = !self.get_register_value(Register::A);
                self.handle_save_register(Register::A, value);
                self.set_flag(FlagBit::AddSub, true);
                self.set_flag(FlagBit::HalfCarry, true);
            }
            Opcode::CCF => {
                self.set_flag(FlagBit::Carry, !self.get_flag(FlagBit::Carry));
                self.set_flag(FlagBit::AddSub, false);
                self.set_flag(FlagBit::HalfCarry, false);
            }
            Opcode::SCF => {
                self.set_flag(FlagBit::Carry, true);
                self.set_flag(FlagBit::AddSub, false);
                self.set_flag(FlagBit::HalfCarry, false);
            }
            Opcode::Halt | Opcode::Stop => {
                self.interrupts.halted = true;
            }
            _ => {
                println!("unhandled opcode {:X?}", opcode);
                panic!();
            }
        }
        if jump_location.is_some() {
            self.program_state.program_counter = jump_location.unwrap();
        }

        if cycle_cost == 0 {
            panic!("Cycle cost not set correctly");
        }
        self.handle_cycle_increment(cycle_cost);
    }

    fn handle_cycle_increment(&mut self, increment: u16) -> () {
        let old_cycle_count = self.program_state.cycle_count;
        self.program_state.cycle_count += increment as u64;
        self.handle_timer(old_cycle_count, self.program_state.cycle_count);
        self.handle_lcd(old_cycle_count, self.program_state.cycle_count);
    }

    fn handle_timer(&mut self, old_count: u64, new_count: u64) -> () {
        // CPU operates at 4.194304Mhz
        // FF04 is Divider register, increments at 16384Hz (every 256 cycles)
        // FF05 is incremented by timer, reset to FF06 when it overflows and interrupt is called
        // FF07 is timer control
        // bit 2 = on/off
        // bit 1-0:
        //   00:   4096Hz (every 1024 cycles)
        //   01: 262144Hz (every 16 cycles)
        //   10:  65536Hz (every 64 cycles)
        //   11:  16384Hz (every 256 cycles)
        let difference = (new_count >> 8) - (old_count >> 8);
        if difference > 0 {
            let divider = self.read_memory(0xFF04);
            self.save_memory(0xFF04, divider.wrapping_add(difference as u8));
        }

        let timer_control = self.read_memory(0xFF07);
        if timer_control & 0x4 != 0 {
            let bit_shift_amount = match timer_control & 0x3 {
                0b00 => 10,
                0b01 => 4,
                0b10 => 6,
                0b11 => 8,
                _ => panic!("invalid timer control"),
            };
            let timer_difference =
                (new_count >> bit_shift_amount) - (old_count >> bit_shift_amount);
            if timer_difference > 0 {
                let value = self.read_memory(0xFF05);
                let (new_value, did_overflow) = value.overflowing_add(timer_difference as u8);
                if did_overflow {
                    self.save_memory(0xFF05, self.read_memory(0xFF06));
                    self.set_interrupt(InterruptBit::Timer);
                } else {
                    self.save_memory(0xFF05, new_value);
                }
            }
        }
    }

    fn handle_lcd(&mut self, old_count: u64, new_count: u64) -> () {
        const LINE_RENDER_CYCLE_COUNT: u64 = 456;
        // CPU operates at 4.194304Mhz
        // V-Blank interrupt at 59.7Hz on a regular GB (every ~70,224 cycles)
        // Increment LY every 456 cycles
        // LY can be 0..=153
        // when LY is 144..=153, we are V-blanking.
        if (new_count / LINE_RENDER_CYCLE_COUNT) > (old_count / LINE_RENDER_CYCLE_COUNT) {
            let old_ly = self.read_memory(constants::LY);
            let new_ly = match old_ly {
                0..=142 | 144..=152 => old_ly + 1,
                143 => {
                    self.set_interrupt(InterruptBit::VBlank);
                    old_ly + 1
                }
                153 => 0,
                154..=0xFF => panic!("unhandled LY value"),
            };
            self._save_memory(constants::LY, new_ly);
            self.do_lyc_compare();
        }

        let current_ly = self.read_memory(constants::LY);
        let new_stat_mode_flag = match current_ly {
            0..=143 => {
                // Line render takes 456 cycles.
                // Mode 2 for 80 cycles
                // Mode 3 for 172 cycles
                // Mode 0 for 204 cycles
                match new_count % LINE_RENDER_CYCLE_COUNT {
                    0..=80 => 2,
                    81..=252 => 3,
                    253..=456 => 0,
                    _ => unreachable!(),
                }
            }
            144..=153 => 1,
            154..=0xFF => panic!("unhandled LY value"),
        };
        let old_stat_mode_flag = self.read_memory(0xFF41) & 0x3;
        if old_stat_mode_flag != new_stat_mode_flag {
            if new_stat_mode_flag == 3 {
                self.do_render_line(self.read_memory(constants::LY));
            } else {
                let bit = match new_stat_mode_flag {
                    0 => LCDCInterruptBit::HBlank,
                    1 => LCDCInterruptBit::VBlank,
                    2 => LCDCInterruptBit::OAM,
                    _ => unreachable!(),
                };
                if self.is_stat_interrupt_enabled(bit) {
                    self.set_interrupt(InterruptBit::LCDStatus)
                }
            }
        }
        self.save_stat_mode_flag(new_stat_mode_flag)
    }

    fn save_stat_mode_flag(&mut self, mode_flag: u8) -> () {
        if mode_flag > 0x3 {
            panic!("Mode flag value {} given for STAT", mode_flag);
        }
        let old_stat = self.read_memory(0xFF41);
        let new_stat = (old_stat & 0xFC) | mode_flag;
        self.memory.other_ram[0xFF41] = new_stat;
    }

    fn save_stat_match_flag(&mut self, matching: bool) -> () {
        let old_stat = self.read_memory(0xFF41);
        let new_stat = (old_stat & 0xFB) | (if matching { 0x4 } else { 0x0 });
        self.memory.other_ram[0xFF41] = new_stat;
    }

    fn do_lyc_compare(&mut self) -> () {
        let ly = self.read_memory(constants::LY);
        let lyc = self.read_memory(0xFF45);
        self.save_stat_match_flag(ly == lyc);

        if self.is_stat_interrupt_enabled(LCDCInterruptBit::LYCLYCoincidence) && ly == lyc {
            self.set_interrupt(InterruptBit::LCDStatus);
        }
    }

    fn is_stat_interrupt_enabled(&self, bit: LCDCInterruptBit) -> bool {
        self.read_memory(0xFF41) & (bit as u8) != 0
    }

    fn set_interrupt(&mut self, bit: InterruptBit) {
        self.interrupts.request_flag = self.interrupts.request_flag | interrupt_picker(bit);
    }

    fn do_math_reg(&mut self, register: Register, f: fn(u8, u8) -> math::Result) -> () {
        self.do_math(self.get_register_value(register), f);
    }

    fn do_bit_op(&mut self, register: Register, f: fn(u8) -> math::Result) -> () {
        let a = self.get_register_value(register);
        let result = f(a);
        self.handle_save_register(register, result.value);
        self.apply_math_result_flags(result);
    }

    fn do_bit_op_carry(&mut self, register: Register, f: fn(u8, bool) -> math::Result) -> () {
        let a = self.get_register_value(register);
        let result = f(a, self.get_flag(FlagBit::Carry));
        self.handle_save_register(register, result.value);
        self.apply_math_result_flags(result);
    }

    fn do_math_carry_reg(&mut self, register: Register, f: fn(u8, u8, bool) -> math::Result) -> () {
        self.do_math_carry(self.get_register_value(register), f);
    }

    fn do_math_carry(&mut self, value: u8, f: fn(u8, u8, bool) -> math::Result) -> () {
        let a = self.get_register_value(Register::A);
        let result = f(a, value, self.get_flag(FlagBit::Carry));
        self.handle_save_register(Register::A, result.value);
        self.apply_math_result_flags(result);
    }

    fn do_math(&mut self, value: u8, f: fn(u8, u8) -> math::Result) -> () {
        let a = self.get_register_value(Register::A);
        let result = f(a, value);
        self.handle_save_register(Register::A, result.value);
        self.apply_math_result_flags(result);
    }

    fn apply_math_result_flags(&mut self, result: math::Result) {
        if result.zero.is_some() {
            self.set_flag(FlagBit::Zero, result.zero.unwrap());
        }
        if result.add_sub.is_some() {
            self.set_flag(FlagBit::AddSub, result.add_sub.unwrap());
        }
        if result.half_carry.is_some() {
            self.set_flag(FlagBit::HalfCarry, result.half_carry.unwrap());
        }
        if result.carry.is_some() {
            self.set_flag(FlagBit::Carry, result.carry.unwrap());
        }
    }

    fn apply_math_result16_flags(&mut self, result: math::Result16) {
        if result.zero.is_some() {
            self.set_flag(FlagBit::Zero, result.zero.unwrap());
        }
        if result.add_sub.is_some() {
            self.set_flag(FlagBit::AddSub, result.add_sub.unwrap());
        }
        if result.half_carry.is_some() {
            self.set_flag(FlagBit::HalfCarry, result.half_carry.unwrap());
        }
        if result.carry.is_some() {
            self.set_flag(FlagBit::Carry, result.carry.unwrap());
        }
    }

    fn do_call(&mut self, address: u16) -> Option<u16> {
        let old_pc = self.program_state.program_counter;
        let old_sp = self.program_state.stack_pointer;
        self.save_memory(old_sp - 1, ((old_pc & 0xff00) >> 8) as u8);
        self.save_memory(old_sp - 2, (old_pc & 0x00ff) as u8);
        self.program_state.stack_pointer = old_sp - 2;
        Some(address)
    }

    fn do_return(&mut self) -> Option<u16> {
        let old_sp = self.program_state.stack_pointer;
        let new_pc: u16 =
            (self.read_memory(old_sp) as u16) + ((self.read_memory(old_sp + 1) as u16) << 8);
        self.program_state.stack_pointer = self.program_state.stack_pointer + 2;
        Some(new_pc)
    }

    fn register_pair_value(&self, hi_register: Register, lo_register: Register) -> u16 {
        let hi_addr = self.get_register_value(hi_register);
        let lo_addr = self.get_register_value(lo_register);
        ((hi_addr as u16) << 8) + (lo_addr as u16)
    }

    fn save_register_pair(
        &mut self,
        hi_register: Register,
        lo_register: Register,
        value: u16,
    ) -> () {
        if hi_register == Register::SPHi && lo_register == Register::SPLo {
            self.program_state.stack_pointer = value;
            return;
        }

        self.handle_save_register(hi_register, ((value & 0xff00) >> 8) as u8);
        self.handle_save_register(lo_register, (value & 0xff) as u8);
    }

    fn get_register_value(&self, register: Register) -> u8 {
        match register {
            Register::A => self.register_state.a,
            Register::B => self.register_state.b,
            Register::C => self.register_state.c,
            Register::D => self.register_state.d,
            Register::E => self.register_state.e,
            Register::F => self.register_state.f,
            Register::H => self.register_state.h,
            Register::L => self.register_state.l,
            Register::SPHi => ((self.program_state.stack_pointer & 0xff00) >> 8) as u8,
            Register::SPLo => (self.program_state.stack_pointer & 0x00ff) as u8,
            Register::PCHi => ((self.program_state.program_counter & 0xff00) >> 8) as u8,
            Register::PCLo => (self.program_state.program_counter & 0x00ff) as u8,
            Register::SpecialLoadHL => {
                self.read_memory(self.register_pair_value(Register::H, Register::L))
            }
            Register::Empty => {
                panic!("unhandled get register value");
            }
        }
    }

    fn handle_save_register(&mut self, register: Register, value: u8) -> () {
        let value = if register == Register::F {
            value & 0xf0
        } else {
            value
        };
        if register == Register::SpecialLoadHL {
            self.save_memory(self.register_pair_value(Register::H, Register::L), value);
        } else {
            let field = match register {
                Register::A => &mut self.register_state.a,
                Register::B => &mut self.register_state.b,
                Register::C => &mut self.register_state.c,
                Register::D => &mut self.register_state.d,
                Register::E => &mut self.register_state.e,
                Register::F => &mut self.register_state.f,
                Register::H => &mut self.register_state.h,
                Register::L => &mut self.register_state.l,
                Register::SPHi
                | Register::SPLo
                | Register::PCHi
                | Register::PCLo
                | Register::SpecialLoadHL
                | Register::Empty => panic!("unhandled save register"),
            };
            *field = value;
        }
    }

    fn set_flag(&mut self, flag: FlagBit, set: bool) -> () {
        let flag_picker = flag_picker(flag);
        let old_value = self.get_register_value(Register::F);
        let new_value = if set {
            old_value | flag_picker
        } else {
            old_value & (!flag_picker)
        };
        self.handle_save_register(Register::F, new_value);
    }

    fn get_flag(&self, flag: FlagBit) -> bool {
        self.get_register_value(Register::F) & flag_picker(flag) != 0
    }

    fn read_memory(&self, address: u16) -> u8 {
        match address {
            0x0000..=0x3FFF => self.rom.read_rom(address as usize),
            0x4000..=0x7FFF => self
                .rom
                .read_rom_bank(self.program_state.rom_bank, address as usize),
            0x8000..=0x9FFF => self.memory.video_ram[(address - 0x8000) as usize],
            0xA000..=0xBFFF => self.memory.external_ram[(address - 0xA000) as usize],
            0xC000..=0xCFFF => self.memory.work_ram_0[(address - 0xC000) as usize],
            0xD000..=0xDFFF => self.memory.work_ram_1[(address - 0xD000) as usize],
            0xE000..=0xFDFF => panic!("unimplemented echo memory"),
            0xFF0F => self.interrupts.request_flag,
            0xFFFF => self.interrupts.enable_flag,
            constants::JOYPAD => {
                self.joypad_value()
                    | (match self.joypad_input.select {
                        JoypadSelectFilter::Undetermined => 0x30,
                        JoypadSelectFilter::Button => 0x10,
                        JoypadSelectFilter::Direction => 0x20,
                    })
                    // GameBoy and GameBoy Pocket return the highest 2 bits as on
                    | 0xC0
            }
            0xFE00..=0xFFFE => self.memory.other_ram[address as usize],
        }
    }

    fn read_memory_bit(&self, address: u16, bit: u8) -> bool {
        assert!(bit <= 7);
        (self.read_memory(address) & (1 << bit)) != 0
    }

    fn cond_memory_bit<T>(&self, address: u16, bit: u8, not_set: T, set: T) -> T {
        if self.read_memory_bit(address, bit) {
            set
        } else {
            not_set
        }
    }

    fn save_memory(&mut self, address: u16, value: u8) -> () {
        let modified_value = match address {
            constants::LY => 0,
            _ => value,
        };
        self._save_memory(address, modified_value)
    }

    fn _save_memory(&mut self, address: u16, value: u8) -> () {
        match address {
            0xFF70 => panic!("writing to ram bank switcher"),
            0x0000..=0x7FFF => match self.rom.cartridge_type() {
                MemoryBankType::MBC1 => match address {
                    0x0000..=0x1FFF => {
                        if (value & 0x0A) != 0 {
                            self.memory.external_ram_enabled = true;
                        } else {
                            self.memory.external_ram_enabled = false;
                        }
                    }
                    0x2000..=0x3FFF => {
                        let bank = (0x1F & value) | (if (value & 0xf) == 0x0 { 0x1 } else { 0x0 });
                        self.program_state.rom_bank = bank;
                    }
                    _ => panic!(
                        "MBC1 writing to rom at {:X} with value {:X}",
                        address, value
                    ),
                },
                MemoryBankType::ROM => match address {
                    0x0000..=0x7FFF => self.rom.write_rom(address as usize, value),
                    _ => panic!(
                        "Memory Bank Type ROM only writing to rom at {:X} with value {:X}",
                        address, value
                    ),
                },
                _ => panic!(
                    "({:?}) writing to rom at {:X} with value {:X}",
                    self.rom.cartridge_type(),
                    address,
                    value
                ),
            },
            0x8000..=0x9FFF => self.memory.video_ram[(address - 0x8000) as usize] = value,
            0xA000..=0xBFFF => {
                if self.memory.external_ram_enabled
                    && self.rom.cartridge_type() != MemoryBankType::ROM
                {
                    self.memory.external_ram[(address - 0xA000) as usize] = value
                } else {
                    panic!("external ram disabled")
                }
            }
            0xC000..=0xCFFF => self.memory.work_ram_0[(address - 0xC000) as usize] = value,
            0xD000..=0xDFFF => self.memory.work_ram_1[(address - 0xD000) as usize] = value,
            0xE000..=0xFDFF => {
                self.save_memory((address - 0xE000) + 0xC000, value);
            }
            0xFF0F => self.interrupts.request_flag = value,
            0xFFFF => self.interrupts.enable_flag = value,
            0xFF41 => {
                let old_stat = self.read_memory(0xFF41);
                let new_stat = (old_stat & 0x7) | (value & 0x78);
                self.memory.other_ram[0xFF41] = new_stat;
            }
            constants::LY | 0xFF45 => {
                self.memory.other_ram[address as usize] = value;
                self.do_lyc_compare();
            }
            0xFF46 => {
                self.memory.other_ram[address as usize] = value;
                let source = (value as u16) << 8;
                for i in 0..0x9F {
                    self.save_memory(0xFE00 + i, self.read_memory(source + i));
                }
            }
            constants::JOYPAD => {
                // TODO: figure out what to do when multiple bits are set
                if value & 0x30 == 0x30 {
                    self.joypad_input.select = JoypadSelectFilter::Undetermined
                } else if value & 0x20 != 0 {
                    self.joypad_input.select = JoypadSelectFilter::Direction
                } else if value & 0x10 != 0 {
                    self.joypad_input.select = JoypadSelectFilter::Button
                }
            }
            0xFF51..=0xFF55 => panic!("cgb vram"),
            0xFE00..=0xFFFE => {
                if address == 0xFF01 {
                    self.output.push(value as char);
                    print!("{}", value as char);
                    io::stdout().flush().ok();
                }
                self.memory.other_ram[address as usize] = value;
            }
        }
    }

    fn set_half_carry_add(&mut self, a: u8, b: u8) -> () {
        self.set_flag(FlagBit::HalfCarry, (((a & 0xf) + (b & 0xf)) & 0x10) == 0x10)
    }

    fn set_half_carry_sub(&mut self, a: u8, b: u8) -> () {
        self.set_flag(FlagBit::HalfCarry, (a & 0xf) < (b & 0xf))
    }

    fn load_address(&self, address: u16) -> u8 {
        self.read_memory(address)
    }

    pub fn run_program(&mut self) -> () {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| loop {
            self.run_single_instruction();
            if self.program_state.cycle_count % 0x100000 == 0 {
                self.debug_oam();
            }
        }));
        if result.is_err() {
            println!("registers: {:X?}", self.register_state);
            println!("program state: {:X?}", self.program_state);
        }
    }

    pub fn safely_run_instruction(&mut self) -> Result<(), ()> {
        let old_pc = self.program_state.program_counter;
        panic::set_hook(Box::new(|_info| {}));
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            self.run_single_instruction();
        }));
        if result.is_err() {
            self.program_state.program_counter = old_pc;
        }
        match result {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }

    pub fn debug_oam(&self) -> () {
        let offset = 0xFE00;
        let mut nonzero = 0;
        for i in 0..40 {
            let pos = i * 4;
            let y = self.read_memory(offset + pos);
            let x = self.read_memory(offset + pos + 1);
            let tile = self.read_memory(offset + pos + 2);
            let attr = self.read_memory(offset + pos + 3);
            if x + y + tile + attr > 0 {
                nonzero += 1;
            }
            //println!("{}, y: {}, x: {}, tile: {}, attr: {:X}", i, y, x, tile, attr);
        }
        for i in 0..1024 {
            let first = self.read_memory(0x9800 + i);
            let second = self.read_memory(0x9C00 + i);
            if first + second > 0 {
                nonzero += 1;
            }
        }
        if nonzero > 0 {
            //println!("nonzero {}", nonzero);
        }

        /*println!("\n\n\n");
        for y in 0..256 {
            for x in 0..256 {
                let (set, shade) = self.screen_output.screen[y * 256 + x];
                print!("{}", shade);
            }
            println!("");
        }*/
    }

    pub fn pixel_at(&self, x: usize, y: usize) -> u8 {
        if x >= 160 || y >= 144 {
            0
        } else {
            self.screen_output.screen[y * 160 + x]
        }
    }

    pub fn do_render_line(&mut self, line: u8) -> () {
        // 0 <= line <= 143

        if !self.read_memory_bit(constants::LCDC, 7) {
            return;
        }

        let mut background_line = vec![0; 160];
        // NOTE: This bit is handled differently in CGB.
        let should_clear_bg = !self.read_memory_bit(constants::LCDC, 0);
        {
            let bg_tile_map = self.cond_memory_bit(constants::LCDC, 3, 0x9800, 0x9C00);
            let tile_start = self.cond_memory_bit(constants::LCDC, 4, 0x8800, 0x8000);
            let signed_tile = !self.read_memory_bit(constants::LCDC, 4);
            let scx = self.read_memory(constants::SCX);
            let scy = self.read_memory(constants::SCY);
            let y = line as usize;
            for x in 0..160 {
                let index: usize = (160 * y) + x;
                let shifted_x = ((scx as usize) + x) % 256;
                let shifted_y = ((scy as usize) + y) % 256;
                let shade = if should_clear_bg {
                    0
                } else {
                    self.bg_shade_at_point(
                        shifted_x as u16,
                        shifted_y as u16,
                        bg_tile_map,
                        tile_start,
                        signed_tile,
                    )
                };
                self.screen_output.screen[index] = shade;
                background_line[x] = shade;
            }
        }

        if self.read_memory_bit(constants::LCDC, 0) && self.read_memory_bit(constants::LCDC, 5) {
            let window_tile_map = self.cond_memory_bit(constants::LCDC, 6, 0x9800, 0x9C00);
            // TODO: Finish window implementation
        }

        // Sprite rendering.
        if self.read_memory_bit(constants::LCDC, 1) {
            let large_size = self.read_memory_bit(constants::LCDC, 2);
            let mut sprites_used = 0;
            for i in 0..40 {
                let y = self.read_memory(constants::SPRITE_TABLE + (i * 4));
                let x = self.read_memory(constants::SPRITE_TABLE + (i * 4) + 1);
                let tile_start = self.read_memory(constants::SPRITE_TABLE + (i * 4) + 2);
                let attributes = self.read_memory(constants::SPRITE_TABLE + (i * 4) + 3);

                // Determine if this sprite affects our line.
                if y == 0
                    || y >= 160
                    || !(y <= line + 16 && line < y)
                    || (!large_size && y <= line + 8)
                    || sprites_used >= 10
                {
                    continue;
                }

                sprites_used = sprites_used + 1;

                // If the sprite is hidden due to its x-coordinate, it is still counted for
                // the purpose of only allowing 10 sprites.
                if x == 0 || x >= 168 {
                    continue;
                }

                let obj_behind_bg = attributes & (1 << 7) != 0;
                let y_flip = attributes & (1 << 6) != 0;
                let x_flip = attributes & (1 << 5) != 0;
                let palette = (attributes & (1 << 4)) >> 4;
                let palette_address = if palette == 0 {
                    constants::SPRITE_PALETTE_0
                } else {
                    constants::SPRITE_PALETTE_1
                };

                let tile_line_to_render = if y_flip { y - line - 1 } else { line + 16 - y };
                let tile_number = if large_size {
                    (tile_start & 0xFE) + if tile_line_to_render > 7 { 1 } else { 0 }
                } else {
                    tile_start
                };
                let tile_address = 0x8000 + ((tile_number as u16) * 16);
                let adjusted_tile_line = (tile_line_to_render & 7) as u16;
                for pixel_x in 0..8 {
                    if x + pixel_x < 8 || x + pixel_x - 8 >= 160 {
                        continue;
                    }

                    let inner_tile_x = if x_flip { 7 - pixel_x } else { pixel_x };
                    let (shade, palette_shade) = self.shade_at_point(
                        tile_address,
                        adjusted_tile_line,
                        inner_tile_x,
                        palette_address,
                    );
                    let final_x = (x + pixel_x - 8) as usize;
                    if shade != 0 && (!obj_behind_bg || background_line[final_x] == 0) {
                        let index: usize = (160 * (line as usize)) + final_x;
                        self.screen_output.screen[index] = palette_shade;
                    }
                }
            }
        }
    }

    // 0 <= x, y <= 256
    fn bg_shade_at_point(
        &self,
        x: u16,
        y: u16,
        bg_tile_map: u16,
        tile_start: u16,
        signed_tile: bool,
    ) -> u8 {
        let tile_y = y / 8;
        let tile_x = x / 8;
        let mut bg_tile = self.read_memory(bg_tile_map + (tile_y * 32) + tile_x);
        if signed_tile {
            bg_tile = (((bg_tile as i8) as i16) + 128) as u8;
        }
        let starting_address = tile_start + ((bg_tile as u16) * 16);
        let inner_tile_x = (x % 8) as u8;
        let inner_tile_y = y % 8;
        let (_shade, palette_shade) = self.shade_at_point(
            starting_address,
            inner_tile_y,
            inner_tile_x,
            constants::BG_PALETTE,
        );
        palette_shade
    }

    fn shade_at_point(
        &self,
        tile_block: u16,
        tile_number: u16,
        tile_x: u8,
        palette_address: u16,
    ) -> (u8, u8) {
        let lsb_data = self.read_memory(tile_block + (tile_number * 2));
        let msb_data = self.read_memory(tile_block + (tile_number * 2) + 1);
        let bit_picker = 1 << (7 - tile_x);
        let shade = (if (msb_data & bit_picker) != 0 {
            0b10
        } else {
            0b0
        }) | (if (lsb_data & bit_picker) != 0 {
            0b1
        } else {
            0b0
        });
        let palette_shade = (self.read_memory(palette_address) >> (shade * 2)) & 0b11;
        (shade, palette_shade)
    }

    fn window_shade_at_point(&self, x: u16, y: u16) -> Option<u8> {
        // 7 <= wx <= 166
        let wx = self.read_memory(constants::WX) as u16;
        // 0 <= wy <= 143
        let wy = self.read_memory(constants::WY) as u16;
        if wy > 143 || wx < 7 || wx > 166 {
            return None;
        }

        if wy > y || wx - 7 > x {
            return None;
        }

        let tile_y = (y - wy) / 8;
        let tile_x = (x + 7 - wx) / 8;

        None
    }

    pub fn push_button(&mut self, button: JoypadButton) -> () {
        if !self.joypad_input.pressed_keys.contains(&button)
            && joypad_button_type(button) == self.joypad_input.select
        {
            self.set_interrupt(InterruptBit::Joypad);
        }
        self.joypad_input.pressed_keys.insert(button);
    }

    pub fn release_button(&mut self, button: JoypadButton) -> () {
        self.joypad_input.pressed_keys.remove(&button);
    }

    fn joypad_value(&self) -> u8 {
        let mut value = 0xF;
        for button in self.joypad_input.pressed_keys.iter() {
            if self.joypad_input.select == JoypadSelectFilter::Button {
                value &= match button {
                    JoypadButton::A => !0x1,
                    JoypadButton::B => !0x2,
                    JoypadButton::Select => !0x4,
                    JoypadButton::Start => !0x8,
                    _ => 0xF,
                }
            } else if self.joypad_input.select == JoypadSelectFilter::Direction {
                value &= match button {
                    JoypadButton::Right => !0x1,
                    JoypadButton::Left => !0x2,
                    JoypadButton::Up => !0x4,
                    JoypadButton::Down => !0x8,
                    _ => 0xF,
                }
            }
        }
        value
    }
}

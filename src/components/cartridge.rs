use std::cmp::Ordering;
use std::fs;
use std::num::Wrapping;

use super::opcode::{nth_register, FlagBit, MemoryBankType, Opcode, Register};

const NINTENDO_LOGO: [u8; 48] = [
    0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B, 0x03, 0x73, 0x00, 0x83, 0x00, 0x0C, 0x00, 0x0D,
    0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E, 0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99,
    0xBB, 0xBB, 0x67, 0x63, 0x6E, 0x0E, 0xEC, 0xCC, 0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E,
];

pub struct ROM {
    content: Vec<u8>,
}

impl ROM {
    pub fn from_path(filename: &str) -> ROM {
        ROM {
            content: match fs::read(filename) {
                Ok(bytes) => bytes,
                Err(_e) => vec![],
            },
        }
    }

    #[cfg(test)]
    pub fn from_bytes(bytes: Vec<u8>) -> ROM {
        ROM { content: bytes }
    }

    pub fn title(&self) -> String {
        let title: Vec<u8> = self.content[0x134..0x144]
            .iter()
            .take_while(|&&v| v > 0)
            .cloned()
            .collect();
        let str = String::from_utf8(title).unwrap();
        str
    }

    pub fn opcode(&self, address: u16, reader: impl Fn(u16) -> u8) -> (Opcode, u16, u16) {
        let immediate8 = || reader(address + 1);
        let relative8 = || immediate8() as i8;
        let immediate16 = || ((reader(address + 2) as u16) << 8) + (reader(address + 1) as u16);
        let opcode_value = reader(address);
        match opcode_value {
            0x00 => (Opcode::Noop, 1, 4),
            0x02 => (
                Opcode::LoadRegisterIntoMemory(Register::A, Register::B, Register::C),
                1,
                8,
            ),
            0x12 => (
                Opcode::LoadRegisterIntoMemory(Register::A, Register::D, Register::E),
                1,
                8,
            ),
            0x08 => (Opcode::SaveSP(immediate16()), 3, 20),
            0x09 => (Opcode::AddHL(Register::B, Register::C), 1, 8),
            0x19 => (Opcode::AddHL(Register::D, Register::E), 1, 8),
            0x29 => (Opcode::AddHL(Register::H, Register::L), 1, 8),
            0x39 => (Opcode::AddHL(Register::SPHi, Register::SPLo), 1, 8),
            0xE8 => (Opcode::AddSP(relative8()), 2, 16),
            0x10 => (Opcode::Stop, 1, 4),
            0x27 => (Opcode::DAA, 1, 4),
            0xC3 => (Opcode::Jump(immediate16()), 3, 16),
            0xE9 => (Opcode::JumpHL, 1, 4),
            0xC2 => (Opcode::JumpCond(FlagBit::Zero, false, immediate16()), 3, 0),
            0xD2 => (Opcode::JumpCond(FlagBit::Carry, false, immediate16()), 3, 0),
            0xCA => (Opcode::JumpCond(FlagBit::Zero, true, immediate16()), 3, 0),
            0xDA => (Opcode::JumpCond(FlagBit::Carry, true, immediate16()), 3, 0),
            0x18 => (Opcode::JumpRelative(relative8()), 2, 12),
            0x20 => (
                Opcode::JumpRelativeCond(FlagBit::Zero, false, relative8()),
                2,
                0,
            ),
            0x28 => (
                Opcode::JumpRelativeCond(FlagBit::Zero, true, relative8()),
                2,
                0,
            ),
            0x30 => (
                Opcode::JumpRelativeCond(FlagBit::Carry, false, relative8()),
                2,
                0,
            ),
            0x38 => (
                Opcode::JumpRelativeCond(FlagBit::Carry, true, relative8()),
                2,
                0,
            ),
            0xF8 => (Opcode::SaveHLSP(relative8()), 2, 12),
            0x01 => (
                Opcode::Load16(Register::B, Register::C, immediate16()),
                3,
                12,
            ),
            0x11 => (
                Opcode::Load16(Register::D, Register::E, immediate16()),
                3,
                12,
            ),
            0x21 => (
                Opcode::Load16(Register::H, Register::L, immediate16()),
                3,
                12,
            ),
            0x31 => (
                Opcode::Load16(Register::SPHi, Register::SPLo, immediate16()),
                3,
                12,
            ),
            0x06 | 0x0E | 0x16 | 0x1E | 0x26 | 0x2E | 0x3E => (
                Opcode::Load8(nth_register((opcode_value & 0x38) >> 3), immediate8()),
                2,
                8,
            ),
            0x36 => (Opcode::Load8(Register::SpecialLoadHL, immediate8()), 2, 12),
            0xF3 => (Opcode::DisableInterrupts, 1, 4),
            0xFB => (Opcode::EnableInterrupts, 1, 4),
            0xEA => (Opcode::SaveRegister(Register::A, immediate16()), 3, 16),
            0xFA => (Opcode::LoadAddress(Register::A, immediate16()), 3, 16),
            0xE0 => (
                Opcode::SaveRegister(Register::A, 0xff00 + (immediate8() as u16)),
                2,
                12,
            ),
            0xF0 => (
                Opcode::LoadAddress(Register::A, 0xff00 + (immediate8() as u16)),
                2,
                12,
            ),
            0xE2 => (Opcode::SaveAIntoRamC, 1, 8),
            0xF2 => (Opcode::LoadRamCIntoA, 1, 8),
            0xC9 => (Opcode::Return, 1, 16),
            0xC0 => (Opcode::ReturnCond(FlagBit::Zero, false), 1, 0),
            0xD0 => (Opcode::ReturnCond(FlagBit::Carry, false), 1, 0),
            0xC8 => (Opcode::ReturnCond(FlagBit::Zero, true), 1, 0),
            0xD8 => (Opcode::ReturnCond(FlagBit::Carry, true), 1, 0),
            0xD9 => (Opcode::ReturnInterrupt, 1, 16),
            0xCD => (Opcode::Call(immediate16()), 3, 24),
            0xC4 => (Opcode::CallCond(FlagBit::Zero, false, immediate16()), 3, 0),
            0xCC => (Opcode::CallCond(FlagBit::Zero, true, immediate16()), 3, 0),
            0xD4 => (Opcode::CallCond(FlagBit::Carry, false, immediate16()), 3, 0),
            0xDC => (Opcode::CallCond(FlagBit::Carry, true, immediate16()), 3, 0),
            0x76 => (Opcode::Halt, 1, 4),
            0x0A => (
                Opcode::LoadAddressFromRegisters(Register::A, Register::B, Register::C),
                1,
                8,
            ),
            0x1A => (
                Opcode::LoadAddressFromRegisters(Register::A, Register::D, Register::E),
                1,
                8,
            ),
            0x40..=0x7F => {
                let right_register = nth_register(opcode_value & 0x7);
                let left_register = nth_register((opcode_value & 0x38) >> 3);
                if right_register == Register::SpecialLoadHL {
                    (
                        Opcode::LoadAddressFromRegisters(left_register, Register::H, Register::L),
                        1,
                        8,
                    )
                } else if left_register == Register::SpecialLoadHL {
                    (
                        Opcode::LoadRegisterIntoMemory(right_register, Register::H, Register::L),
                        1,
                        8,
                    )
                } else {
                    (Opcode::LoadReg(left_register, right_register), 1, 4)
                }
            }
            0xC5 => (Opcode::Push(Register::B, Register::C), 1, 16),
            0xC1 => (Opcode::Pop(Register::B, Register::C), 1, 12),
            0xD5 => (Opcode::Push(Register::D, Register::E), 1, 16),
            0xD1 => (Opcode::Pop(Register::D, Register::E), 1, 12),
            0xE5 => (Opcode::Push(Register::H, Register::L), 1, 16),
            0xE1 => (Opcode::Pop(Register::H, Register::L), 1, 12),
            0xF5 => (Opcode::Push(Register::A, Register::F), 1, 16),
            0xF1 => (Opcode::Pop(Register::A, Register::F), 1, 12),
            0x03 => (Opcode::IncPair(Register::B, Register::C), 1, 8),
            0x13 => (Opcode::IncPair(Register::D, Register::E), 1, 8),
            0x23 => (Opcode::IncPair(Register::H, Register::L), 1, 8),
            0x33 => (Opcode::IncPair(Register::SPHi, Register::SPLo), 1, 8),
            0x0B => (Opcode::DecPair(Register::B, Register::C), 1, 8),
            0x1B => (Opcode::DecPair(Register::D, Register::E), 1, 8),
            0x2B => (Opcode::DecPair(Register::H, Register::L), 1, 8),
            0x3B => (Opcode::DecPair(Register::SPHi, Register::SPLo), 1, 8),
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x3C => {
                (Opcode::Inc(nth_register((opcode_value & 0x38) >> 3)), 1, 4)
            }
            0x34 => (Opcode::Inc(Register::SpecialLoadHL), 1, 12),
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x3D => {
                (Opcode::Dec(nth_register((opcode_value & 0x38) >> 3)), 1, 4)
            }
            0x35 => (Opcode::Dec(Register::SpecialLoadHL), 1, 12),
            0x22 => (Opcode::SaveHLInc, 1, 8),
            0x2A => (Opcode::LoadHLInc, 1, 8),
            0x32 => (Opcode::SaveHLDec, 1, 8),
            0x3A => (Opcode::LoadHLDec, 1, 8),
            0xA0..=0xA5 | 0xA7 => (Opcode::And(nth_register(opcode_value & 0x7)), 1, 4),
            0xA6 => (Opcode::And(Register::SpecialLoadHL), 1, 8),
            0xA8..=0xAD | 0xAF => (Opcode::Xor(nth_register(opcode_value & 0x7)), 1, 4),
            0xAE => (Opcode::Xor(Register::SpecialLoadHL), 1, 8),
            0xB0..=0xB5 | 0xB7 => (Opcode::Or(nth_register(opcode_value & 0x7)), 1, 4),
            0xB6 => (Opcode::Or(Register::SpecialLoadHL), 1, 8),
            0xB8..=0xBD | 0xBF => (Opcode::Cp(nth_register(opcode_value & 0x7)), 1, 4),
            0xBE => (Opcode::Cp(Register::SpecialLoadHL), 1, 8),
            0xE6 => (Opcode::AndValue(immediate8()), 2, 8),
            0xEE => (Opcode::XorValue(immediate8()), 2, 8),
            0xF6 => (Opcode::OrValue(immediate8()), 2, 8),
            0xFE => (Opcode::CpValue(immediate8()), 2, 8),
            0x80..=0x85 | 0x87 => (Opcode::Add(nth_register(opcode_value & 0x7)), 1, 4),
            0x86 => (Opcode::Add(Register::SpecialLoadHL), 1, 8),
            0x88..=0x8D | 0x8F => (Opcode::AddCarry(nth_register(opcode_value & 0x7)), 1, 4),
            0x8E => (Opcode::AddCarry(Register::SpecialLoadHL), 1, 8),
            0x90..=0x95 | 0x97 => (Opcode::Sub(nth_register(opcode_value & 0x7)), 1, 4),
            0x96 => (Opcode::Sub(Register::SpecialLoadHL), 1, 8),
            0x98..=0x9D | 0x9F => (Opcode::SubCarry(nth_register(opcode_value & 0x7)), 1, 4),
            0x9E => (Opcode::SubCarry(Register::SpecialLoadHL), 1, 8),
            0xC6 => (Opcode::AddValue(immediate8()), 2, 8),
            0xCE => (Opcode::AddCarryValue(immediate8()), 2, 8),
            0xD6 => (Opcode::SubValue(immediate8()), 2, 8),
            0xDE => (Opcode::SubCarryValue(immediate8()), 2, 8),
            0xF9 => (Opcode::LoadHLIntoSP, 1, 8),
            0x07 => (Opcode::RLCA, 1, 4),
            0x0F => (Opcode::RRCA, 1, 4),
            0x17 => (Opcode::RLA, 1, 4),
            0x1F => (Opcode::RRA, 1, 4),
            0x2F => (Opcode::CPL, 1, 4),
            0x3F => (Opcode::CCF, 1, 4),
            0x37 => (Opcode::SCF, 1, 4),
            0xCB => {
                let cb_instr = immediate8();
                let cycle_count = if (cb_instr & 0x7) == 0x6 {
                    if cb_instr >= 0x40 && cb_instr <= 0x7F {
                        12
                    } else {
                        16
                    }
                } else {
                    8
                };
                (self.cb_opcode(cb_instr), 2, cycle_count)
            }
            0xC7 | 0xCF | 0xD7 | 0xDF | 0xE7 | 0xEF | 0xF7 | 0xFF => {
                (Opcode::Restart((opcode_value & 0x38) as u16), 1, 16)
            }
            0xD3 | 0xDB | 0xDD | 0xE3 | 0xE4 | 0xEB | 0xEC | 0xED | 0xF4 | 0xFC | 0xFD => {
                (Opcode::UnimplementedOpcode(opcode_value), 1, 0)
            }
        }
    }

    pub fn cb_opcode(&self, value: u8) -> Opcode {
        match value {
            0x00..=0x07 => Opcode::RLC(nth_register(value & 0x7)),
            0x08..=0x0F => Opcode::RRC(nth_register(value & 0x7)),
            0x10..=0x17 => Opcode::RL(nth_register(value & 0x7)),
            0x18..=0x1F => Opcode::RR(nth_register(value & 0x7)),
            0x20..=0x27 => Opcode::SLA(nth_register(value & 0x7)),
            0x28..=0x2F => Opcode::SRA(nth_register(value & 0x7)),
            0x30..=0x37 => Opcode::Swap(nth_register(value & 0x7)),
            0x38..=0x3F => Opcode::SRL(nth_register(value & 0x7)),
            0x40..=0x7F => Opcode::Bit(nth_register(value & 0x7), (value & 0x38) >> 3),
            0x80..=0xBF => Opcode::Reset(nth_register(value & 0x7), (value & 0x38) >> 3),
            0xC0..=0xFF => Opcode::Set(nth_register(value & 0x7), (value & 0x38) >> 3),
        }
    }

    pub fn has_nintendo_logo(&self) -> bool {
        self.content[0x104..0x134].iter().cmp(NINTENDO_LOGO.iter()) == Ordering::Equal
    }

    pub fn has_valid_header_checksum(&self) -> bool {
        let checksum: Wrapping<u8> = self.content[0x134..0x14D]
            .iter()
            .cloned()
            .map(|v| Wrapping(v))
            .fold(Wrapping(0), |acc, v| acc - v - Wrapping(1));
        checksum.0 == self.content[0x14D]
    }

    pub fn cartridge_type(&self) -> MemoryBankType {
        match self.content[0x147] {
            0x00 | 0x08..=0x09 => MemoryBankType::ROM,
            0x01..=0x03 => MemoryBankType::MBC1,
            0x05..=0x06 => MemoryBankType::MBC2,
            0x0B..=0x0D => MemoryBankType::MMM01,
            0x0F..=0x13 => MemoryBankType::MBC3,
            0x15..=0x17 => MemoryBankType::MBC4,
            0x19..=0x1E => MemoryBankType::MBC5,
            _ => panic!("unknown memory bank type"),
        }
    }

    pub fn ram_size(&self) -> usize {
        match self.content[0x149] {
            0x00 => 0,
            0x01 => 2 * 1024,
            0x02 => 8 * 1024,
            0x03 => 32 * 1024,
            _ => panic!("unknown ram size"),
        }
    }

    pub fn read_rom(&self, address: usize) -> u8 {
        self.content[address]
    }

    pub fn read_rom_bank(&self, bank: u8, address: usize) -> u8 {
        self.content[(0x4000 * (bank as usize)) + address]
    }

    pub fn write_rom(&mut self, address: usize, value: u8) -> () {
        self.content[address] = value
    }
}

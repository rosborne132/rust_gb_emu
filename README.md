# gb_emu

Example invocations
```
cargo run test-roms/blargg/cpu_instrs/individual/01-special.gb --show
cargo run test-roms/kirbo_dream_land.gb --show
cargo run test-roms/dr_mario.gb --show
```

Setup environment
```
sudo apt-get install libncurses5-dev libncursesw5-dev
```

Debugger commands:
- b *number*: run until PC == number
- c: run until error
- x *number*: run for number cycles
- o *number*: run number opcodes

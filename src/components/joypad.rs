// #[derive(Copy, Clone, PartialEq, Eq, Hash)]
// pub enum JoypadButton {
//     Down,
//     Up,
//     Left,
//     Right,
//     A,
//     B,
//     Start,
//     Select,
// }

// fn joypad_button_type(button: JoypadButton) -> JoypadSelectFilter {
//     match button {
//         JoypadButton::Down => JoypadSelectFilter::Direction,
//         JoypadButton::Up => JoypadSelectFilter::Direction,
//         JoypadButton::Left => JoypadSelectFilter::Direction,
//         JoypadButton::Right => JoypadSelectFilter::Direction,
//         JoypadButton::A => JoypadSelectFilter::Button,
//         JoypadButton::B => JoypadSelectFilter::Button,
//         JoypadButton::Start => JoypadSelectFilter::Button,
//         JoypadButton::Select => JoypadSelectFilter::Button,
//     }
// }

// #[derive(PartialEq, Eq, Hash)]
// pub enum JoypadSelectFilter {
//     Undetermined,
//     Button,
//     Direction,
// }

// pub struct JoypadInput {
//     pressed_keys: HashSet<JoypadButton>,
//     select: JoypadSelectFilter,
// }

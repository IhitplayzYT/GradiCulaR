use crate::helper::Helper::CLI;

mod helper;
mod random;
mod tensor;

fn main() {
    let mut clargs = CLI::new();
    clargs.Parse_Args();

    if clargs.dbg{
        println!("{clargs:?}");
    }





}

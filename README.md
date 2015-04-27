# FlatBuffers in Rust

This library provides runtime support for FlatBuffers in Rust.

## Obtaining the Modded FlatBuffers Compiler

I have created a modified version of the flatbuffers compiler which will produce Rust code as
output. It is available in the `rust-gen` branch if my `flatbuffers` repository. It can compiled and
installed via the following command sequence.

```bash
    # Clone the repository
    git clone https://github.com/arbitrary-cat/flatbuffers && cd flatbuffers

    # Checkout the branch with the rust code in it
    git checkout rust-gen

    # Prepare the makefiles with CMake
    cmake .

    # Build the project
    make

    # Install the binaries
    sudo make install
```

Then you can produce `buffers.rs` from `buffers.fbs` by running:

```bash
    flatc -r buffers.fbs
```

## Usage

Once you've produced your `*.rs` files from the `*.fbs` files, move them to the appropriate location
in your Rust program's source tree, and add the corresponding `mod` definitions (just like adding
any other module to a Rust program).

The generated files expect you to have the `num` crate and my `flatbuffers` crate available. You can
accomplish this by adding the following to your `Cargo.toml`

```toml
[dependencies]
num = "0.1.24"

[dependencies.flatbuffers]
git = "https://github.com/arbitrary-cat/flatbuffers-rs"
```

Then you need to add the correct crate imports to the top-level of your crate.

```rust
extern crate flatbuffers;
extern crate num;
```

# starlit one
First generation CV (Computer Vision) intelligent Robot

## Description

This repo is used to publicly showcase and research how I can create an intelligent robot myself, and it includes the following features:

1. Design the robotic arm, including drawing information.
2. Find a suitable development board for the robotic arm, currently I plan to use Jetson.
3. Build the development environment on the development board.

   **Note:** I plan to use Java as the computer vision, [TensorFlow] is my preferred framework. and then I plan to use the Rust language for the [embedded] part
   
   [TensorFlow]: https://github.com/tensorflow
   [embedded]: https://github.com/rust-embedded
   
   the mechanical Motion planning part (I don't think much about C++ and C language) because I have not been exposed to it before.
   
   Compared with these two languages, I consider the safety of the mechanical part, and choose Rust is the best choice.
4. Make a simple object recognition model, I will annotate some data, train this model, and deploy it for testing.
5. Identified data is transmitted to Rust Lib through JNI, Use the craft provided by Rust to interact with the robotic arm and control it

## License

Licensed under either of

 * [MIT License](https://github.com/Jzow/starlat/blob/master/LICENSE-MIT)
 
## Contribution

So far, this repo is only a preliminary beginning, and I will continue to improve it through the knowledge I have learned. If you have any interests and ideas, please submit to PR or contact jameszow@163.com.

We're looking for various contributions!

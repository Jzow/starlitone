use tensorflow::Tensor;
use tensorflow::DataType;
use tensorflow::Shape;

fn x() -> Tensor<f32>{
    let mut numbers = Tensor::<f32>::new(&[12]);

    numbers.set(&[0], 0f32);
    numbers.set(&[1], 1f32);
    numbers.set(&[2], 2f32);
    numbers.set(&[3], 3f32);
    numbers.set(&[4], 4f32);
    numbers.set(&[5], 5f32);
    numbers.set(&[6], 6f32);
    numbers.set(&[7], 7f32);
    numbers.set(&[8], 8f32);
    numbers.set(&[9], 9f32);
    numbers.set(&[10], 10f32);
    // 简写
    numbers[11] = 11.0;

    numbers
}

fn shape(tensor: Tensor<f32>) -> Shape {
    let reshape: Shape = Tensor::shape(&tensor);

    reshape
}



#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_numbers(){
        let result = x();
        println!("{:?}", result);
        println!("size: {:?}", result.len());
    }

    #[test]
    fn test_shape(){
        let result = x();
        println!("{:?}", &result);
        println!("size: {:?}", &result.len());


        let shapes = shape(result);
        println!("shape: {:?}", shapes);
    }

    #[test]
    fn tensor_zero(){
        let zero:Tensor<f64> = Tensor::new(&[2, 3, 4]);
        
        println!("zero tensor: {:?}", zero);
    }
}
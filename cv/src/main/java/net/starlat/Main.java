package net.starlat;

import com.google.protobuf.Int32Value;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.IntNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;

public class Main {
    public static void main(String[] args) {
        tensorFlowSample();
    }

    private static void tensorFlowSample() {
        System.out.println(TensorFlow.version());

        var rank1Tensor = TInt64.vectorOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println(rank1Tensor.shape());

        try (ConcreteFunction dbl = ConcreteFunction.create(Main::dbl);
            TInt32 x = TInt32.scalarOf(10);
            Tensor dblx = dbl.call(x)) {
            System.out.println(x.getInt() + " doubled is " + ((TInt32) dblx).getInt());

            var ndArrays = NdArrays.vectorOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

            NdArrays.ofObjects(TInt32.class, ndArrays.shape());

            System.out.println(ndArrays.scalars());
            System.out.println(ndArrays.shape().isMatrix());

            Shape matrix2 = Shape.of(1, 2, 3, 4);
            Shape batch = Shape.of(-1, 4);
            System.out.println(matrix2);

            FloatNdArray matrix3d = NdArrays.ofFloats(matrix2);

            // Initialize sub-matrices data with vectors
            matrix3d.set(NdArrays.vectorOf(1.0f, 2.0f), 0, 0)
                    .set(NdArrays.vectorOf(3.0f, 4.0f), 0, 1)
                    .set(NdArrays.vectorOf(5.0f, 6.0f), 0, 2)
                    .set(NdArrays.vectorOf(7.0f, 8.0f), 1, 0)
                    .set(NdArrays.vectorOf(9.0f, 10.0f), 1, 1)
                    .set(NdArrays.vectorOf(11.0f, 12.0f), 1, 2);

            // Access the second 3x2 matrix (of rank 2)

            System.out.println(matrix3d);

        }
    }

    private static Signature dbl(Ops tf) {
        Placeholder<TInt32> x = tf.placeholder(TInt32.class);
        Add<TInt32> dblx = tf.math.add(x, x);
        return Signature.builder().input("x", x).output("dbl", dblx).build();
    }
}
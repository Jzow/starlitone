package net.starlat;

import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
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
            System.out.println(x.getInt() + "doubled is " + ((TInt32) dblx).getInt());
        }
    }

    private static Signature dbl(Ops tf) {
        Placeholder<TInt32> x = tf.placeholder(TInt32.class);
        Add<TInt32> dblx = tf.math.add(x, x);
        return Signature.builder().input("x", x).output("dbl", dblx).build();
    }
}
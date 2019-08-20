package com.mccorby.openmined.worker.framework

import com.mccorby.openmined.worker.domain.MLFramework
import com.mccorby.openmined.worker.domain.SyftOperand

class TensorFlowFramework : MLFramework {

    override fun multiply(tensor1: SyftOperand.SyftTensor, tensor2: SyftOperand.SyftTensor): SyftOperand.SyftTensor {
        TODO("not implemented")
    }

    override fun add(tensor1: SyftOperand.SyftTensor, tensor2: SyftOperand.SyftTensor): SyftOperand.SyftTensor {
        TODO("not implemented")
        // // Let's say graph is an instance of the Graph class
        // // for the computation y = 3 * x
        //
        // try (Session s = new Session(graph)) {
        //   try (Tensor x = Tensor.create(2.0f);
        //       Tensor y = s.runner().feed("x", x).fetch("y").run().get(0)) {
        //       System.out.println(y.floatValue());  // Will print 6.0f
        //   }
        //   try (Tensor x = Tensor.create(1.1f);
        //       Tensor y = s.runner().feed("x", x).fetch("y").run().get(0)) {
        //       System.out.println(y.floatValue());  // Will print 3.3f
        //   }
        // }
    }
}
const tf = require('@tensorflow/tfjs-node-gpu');

async function testGPU() {
    await tf.ready();
    console.log('Backend:', tf.getBackend());
    console.log('Memory:', tf.memory());
    
    // Create a simple tensor on GPU
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const result = a.matMul(b);
    
    console.log('Result:', await result.data());
    console.log('GPU test completed!');
}

testGPU();
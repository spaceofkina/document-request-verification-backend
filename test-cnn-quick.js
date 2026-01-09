// test-cnn-quick.js - Quick CNN Test with 11 samples
const tf = require('@tensorflow/tfjs-node');  // Use CPU-optimized version
const sharp = require('sharp');
const path = require('path');
const fs = require('fs').promises;

async function quickTest() {
    console.log('🚀 QUICK TEST: Training CNN with minimal samples\n');
    
    // Simple model (smaller, faster)
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [100, 100, 3],  // Smaller images = faster
        filters: 8,  // Fewer filters = faster
        kernelSize: 3,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 5, activation: 'softmax' }));
    
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    console.log('✅ Tiny CNN model created (for testing)');
    
    // Create synthetic test data (11 samples)
    console.log('📊 Generating 11 synthetic training samples...');
    
    // Create simple colored images as test data
    const xs = [];
    const ys = [];
    
    // 5 document types, 2-3 samples each = 11 total
    for (let i = 0; i < 11; i++) {
        const docType = i % 5;  // 5 document types
        const color = docType * 50;  // Different color per doc type
        
        // Create simple tensor (100x100 RGB image)
        const data = new Array(100 * 100 * 3).fill(0).map((_, idx) => {
            const channel = idx % 3;
            return channel === docType % 3 ? color : 0;
        });
        
        const tensor = tf.tensor3d(data, [100, 100, 3], 'float32').div(255.0);
        xs.push(tensor);
        ys.push(docType);
    }
    
    const xTrain = tf.stack(xs);
    const yTrain = tf.oneHot(tf.tensor1d(ys, 'int32'), 5);
    
    console.log('🏋️ Training with 11 samples (3 seconds expected)...');
    const startTime = Date.now();
    
    // Train for just 3 epochs
    const history = await model.fit(xTrain, yTrain, {
        epochs: 3,
        batchSize: 4,
        verbose: 1,
        validationSplit: 0.2
    });
    
    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;
    
    console.log(`\n✅ QUICK TEST COMPLETE in ${trainingTime.toFixed(1)} seconds!`);
    console.log(`📈 Final accuracy: ${(history.history.acc[2] * 100).toFixed(1)}%`);
    
    // Test prediction
    const testSample = xs[0].expandDims(0);
    const prediction = model.predict(testSample);
    const predData = await prediction.data();
    
    console.log('\n🔍 Sample prediction:');
    console.log(`   Document probabilities:`);
    ['Drivers License', 'Passport', 'National ID', 'Student ID', 'Barangay ID']
        .forEach((doc, idx) => {
            console.log(`   • ${doc}: ${(predData[idx] * 100).toFixed(1)}%`);
        });
    
    // Cleanup
    xTrain.dispose();
    yTrain.dispose();
    xs.forEach(t => t.dispose());
    testSample.dispose();
    prediction.dispose();
    
    console.log('\n🎯 CNN IS WORKING! Ready for full training.');
    
    return {
        success: true,
        trainingTime: trainingTime,
        accuracy: history.history.acc[2],
        note: 'CNN successfully trained with 11 samples'
    };
}

// Run test
quickTest().catch(console.error);
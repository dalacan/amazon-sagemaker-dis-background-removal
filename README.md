# Amazon Ssagemaker DIS background removal

In this this repository, we will go through the steps to implement the DIS background removal model in Amazon SageMaker.

---

In the `dis-background-removal-sagemaker-inference.ipynb` notebook, we will implement asynchronous inferencing using the DIS background removal tool in Amazon SageMaker. By using Amazon SageMaker Asynchronous inference, you can process a large batch of images at scale with large payload sizes (up to 1GB) with near real-time latency.

We will walkthrough on how to package a pre-train DIS model, implement the inferencing with a custom inference script and deploy out asynchronous endpoint.

High Level Steps

Package inference script, model and pre-train weights
Create SNS topics (optional) to record succcess/error messages from asynchronous inference
Create asynchronous configuration and deploy endpoint
Run Inference
# Project-Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker
Building an image classification model that can automatically detect which kind of vehicle delivery drivers have

The CIFAR-100 dataset, hosted by the University of Toronto at https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz, was obtained and transformed into a usable shape and format of 32X32X3 [1]. The images of bicycles and motorcycles were then filtered from the dataset and stored into two dataframes, train and test [2]. These dataframes were used to retrieve the images from the source dataset and load them into AWS S3. A training job was created by retrieving an image-classification model from AWS using a single ml.p3.2xlarge instance [3]. The trained model trained was then deployed into an AWS endpoint with a model monitor to track the deployment [4]. Serverless workflows were orchestrated using Step Functions and three Lambda Functions [5]. The first Lambda Function was responsible for copying an object from S3, encoding it, and returning it to the Step Function as image_data in an event [5.1]. The second Lambda Function decoded the image output from the previous function and passed inferences back to the Step Function [5.2]. This Lambda function had runtime dependencies that needed to be packaged and uploaded as a zip file using CLI commands to the Lambda service. The last Lambda Function was responsible for filtering low-confidence inferences < .093 [5.3]. The final step was to test and evaluate the model's predictions, as shown by the plots that display the model scores [6] and [7].

1. Image transformed

![Example image transformerd](https://github.com/punkmic/Project-Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker/blob/master/output_17_0.png)

2. Test 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filenames</th>
      <th>labels</th>
      <th>row</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>safety_bike_s_000390.png</td>
      <td>8</td>
      <td>27</td>
    </tr>
    <tr>
      <th>28</th>
      <td>bike_s_000658.png</td>
      <td>8</td>
      <td>28</td>
    </tr>
    <tr>
      <th>116</th>
      <td>velocipede_s_001744.png</td>
      <td>8</td>
      <td>116</td>
    </tr>
    <tr>
      <th>161</th>
      <td>bike_s_000643.png</td>
      <td>8</td>
      <td>161</td>
    </tr>
    <tr>
      <th>319</th>
      <td>ordinary_bicycle_s_000437.png</td>
      <td>8</td>
      <td>319</td>
    </tr>
  </tbody>
  </table>
</div>
  
2.1 Train 
  <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filenames</th>
      <th>labels</th>
      <th>row</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>bike_s_000682.png</td>
      <td>8</td>
      <td>16</td>
    </tr>
    <tr>
      <th>30</th>
      <td>bike_s_000127.png</td>
      <td>8</td>
      <td>30</td>
    </tr>
    <tr>
      <th>130</th>
      <td>cycle_s_002598.png</td>
      <td>8</td>
      <td>130</td>
    </tr>
    <tr>
      <th>152</th>
      <td>minibike_s_000824.png</td>
      <td>48</td>
      <td>152</td>
    </tr>
    <tr>
      <th>195</th>
      <td>motorcycle_s_001856.png</td>
      <td>48</td>
      <td>195</td>
    </tr>
  </tbody>
  </table>
</div>


3 Get Model container image and define the model s3 output location

```python
from sagemaker import image_uris
algo_image = image_uris.retrieve(framework='image-classification', region=boto3.Session().region_name)
s3_output_location = f"s3://{bucket}/models/image_model"
```

 3.1 Define a Estimator with one instance of type 'ml.p3.2xlarge' and set the s3 output location

```python
img_classifier_model=sagemaker.estimator.Estimator(
    image_uri=algo_image,
    role = get_execution_role(),
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session()
)
```

 3.2 Set hyperparameters

```python
img_classifier_model.set_hyperparameters(
    image_shape='3,32,32', # ’num_channels, height, width’
    num_classes=len(set(df_train.labels)), # the number of output classes
    num_training_samples= len(df_train) # the total number of training samples
)
```
 3.3 Define four TrainingInput classes

```python
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput
model_inputs = {
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/train/",
            content_type="application/x-image"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/test/",
            content_type="application/x-image"
        ),
        "train_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/train.lst",
            content_type="application/x-image"
        ),
        "validation_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/test.lst",
            content_type="application/x-image"
        )
}
```

 3.4 Fit the model

```python
img_classifier_model.fit(inputs=model_inputs)
```

4. Model deployment

```python
from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    ## TODO: Set config options
    enable_capture=True,
    sampling_percentage=40,
    destination_s3_uri=f"s3://{bucket}/data_capture"
)
```

```python
deployment = img_classifier_model.deploy(
    ## TODO: fill in deployment options
    data_capture_config=data_capture_config,
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
    )
```

5. Step function Worflow

Step Functions            |  Step Functions After running
:-------------------------:|:-------------------------:
![Structure](https://github.com/punkmic/Project-Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker/blob/master/stepfunctions_graph%20(5).png)  |  ![Result](https://github.com/punkmic/Project-Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker/blob/master/stepfunctions_graph%20(7).png)

 5.1 Lambda Function: copy an object from S3 and encode it

```python
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = s3_address = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }
```

 5.2 Decode the image output from the previous function and return the inferences back to the Step Function

```python
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-01-19-16-05-56-860'

def lambda_handler(event, context):
    # Decode the image data
    img = event["image_data"]
    image = base64.b64decode()

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT, sagemaker_session=sagemaker.Session())

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(data=image)
    
    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```

 5.3  Filter low-confidence inferences < .093

```python
def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event["inferences"]
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(float(x) > THRESHOLD for x in inferences.encode('utf-8'))
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```

6. Observed Recent Inferences

<img src= "https://github.com/punkmic/Project-Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker/blob/master/output_84_0.png" width="500" height="500">

7. Prediction Scores over Time

<img src= "https://github.com/punkmic/Project-Build-a-ML-Workflow-For-Scones-Unlimited-On-Amazon-SageMaker/blob/master/Output-98.PNG" width="500" height="500">

**Behavioral Cloning** 

**Behavioral Cloning Project**

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model.mp4 containing the compiled video of a successfull autonomous run.
* writeup_report.md summarizing the results


2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.py file contains all of the code used for training and saving the model. The only changes made to drive.py were preprocessing steps. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.  

Model Architecture and Training Strategy

1. An appropriate model architecture has been employed

The model employs strided convolutional layers followed by an equal number of fully connected layers. It uses Mean Squared Error to determine loss and the adam optimizer.  I employ LeakyRelu on this model to control for dead neurons common to Relu models.  


2. Attempts to reduce overfitting in the model

The dataset was split into training, validation and testing sets immediately after it was assembled to control for overfitting. In addition, the model was intentionally built to only accept grayscale images to reduce overfitting to lighting conditions or color cues.  
Dropout was not used as overfitting did not become visible in the validation set until epoch counts approaching 100. In addition, testing with dropout significantly reduced the ability of the model to converge. By keeping the number of epochs down, I settled on 40, overfitting was minimized.  

Finally, the model was used in the driving simulator to ensure it could keep the vehicle on the track.  


3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 284). I tuned filter size and depth on each convolutional layer. I also tuned the size of the fully connected layers. Finally I adjusted the initialization method to accelerate time to convergence.  


4. Appropriate training data

The training data was gathered in several patterns. I performed a single lap forward at less than 5 mph followed by a u-turn while recording was stopped and recording a lap going the other direction at the same speed. Both of these laps focused on center lane driving.  

Next I included the dataset provided by Udacity, after removing the headers from the csv for consistency.  

Finally, I simulated recovery situations, starting recording with the steering angle already set to return back to the center of the lane.  

As most of this driving was close to straight ahead, the data created a bias to drive straight ahead in the model. By reducing the number samples with steering values of 0.2 to -0.2 (5 degrees to -5 degrees) by 2/3rds, I was able to compensate for this issue.


Model Architecture and Training Strategy

1. Solution Design Approach

The model was built iteratively from a basic single convolution and single fully connected layer. I chose 5 architypical images from the full dataset. One image was chose for each of the values -0.5, -0.25, 0, 0.25 and 0.5. I then trained and validated with only these 5 images until my model had completely memorized these samples.  

It was in this quick validation environment that I added and tuned layers. I discarded AveragePooling and MaxPooling as well as Dropout as detrimental during this testing. By adjusting layer parameters, I was able to bring my loss on this tiny dataset down to approximately 1.5 * 10^-11. As close to effectively zero as I cared to pursue.  

The model was then trained on the full training dataset and validated with the full validation set. I was struck by the speed with which it converged on the full data.  

At this point I began running models on the simulator, I found that below 20 epochs, the model was chaotic and inconsistent from frame to frame. On the other hand, when I went to 50 epochs, the model tended to drive smoothly directly into the the nearest barrier.  

I adjusted the batch size and epochs to reach the final simulator performance. Once I had an acceptable model, I made further attempts to reintroduce Pooling and Dropout layers, they still underperformed.  

With the final configuration of batch size 128 and 40 epochs, the model stays on the road and navigates the course at 30 mph without issue.  


2. Final Model Architecture

My model consists of 3 convolutions followed by 3 fully connected layers with LeakyRelu activations separating the layers (lines 271 - 283). I chose to use glorot normal initialization as it provided the quickest convergence in my testing.  

Each convolution consists of a 3x3 filter on a 2x2 stride. The convolutional filters are set with steadily increasing depth. I started with a filter depth of 12, followed by 16 and finally 24.  

The fully connected layers step down 3 times, from 1176 to 1000, from 1000 to 100 and from 100 to 1 for the final steering output.  


3. Creation of the Training Set & Training Process

As discussed above, I used several methods to gather simulated data. This data was then compiled into a single dataset, irrespective of source. In an effort to maximize the value of the gathered data, for every timestep of the raw datasets, I included the center camera with the recorded steering value, then flipped the center camera over the vertical axis and included that with the opposite steering value.  I used a 0.2 steering offset for the left image and -0.2 for the right. These were also flipped. This resulted in about 58,000 images.  

Filtering that dataset reduced the density of the data with steering values from -0.2 to 0.2 (-5 degrees to 5 degrees), resulting in a final dataset of 27,293 samples. This more balanced dataset was what I split into training, validation and testing using numpy's random choice function.  

The final model reflected a testing loss of 0.0279.  

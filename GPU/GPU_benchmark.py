# # Python program to explain os.environ object  
  
# # importing os module  
# import os 
# import pprint 
  
# # Get the list of user's 
# # environment variables 
# env_var = os.environ 
  
# # Print the list of user's 
# # environment variables 
# print("User's Environment variable:") 
# pprint.pprint(dict(env_var), width = 1) 
# exit()

import tensorflow as tf
print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))
exit()

from ai_benchmark import AIBenchmark
benchmark = AIBenchmark()
results = benchmark.run()
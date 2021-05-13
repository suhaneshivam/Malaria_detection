# MALARIA DETECTOR
Malaria causes approximately 400,000 deaths across the globe per year. This is an
endemic in poverty-stricken areas like Africa and some parts on Asian-subcontinent. The traditional methods to diagnose the MALARIA in a cell is both tedious and time consuming. The person has to manually count the number of infected red blood cells in the blood smears. According to the official WHO malaria parasite counting protocol, a clinician may have to manually count up to 5,000 cells, an extremely tedious and time-consuming process.
Another technique called antigen testing for Rapid Diagnosis Testing (RDT) are significantly faster than cell counting they are also much less accurate.

##### An ideal solution would, therefore, need to combine the _speed_ of RDTs with the _accuracy_ of microscopy.
####Deep Learning is _SOLUTION_!

We trained ResNet-50 architecture on infected and non-infected blood smears. The training took a total of 1 hour to achieve an over-all accuracy of 96%! Model is reliable and can be used for the actual detection of Malaria in real time.

# DATASET
The dataset can be found on the [official NIH website](https://ceb.nlm.nih.gov/repositories/malaria-datasets/ "MALARIA DATASET")
The number of images per class is equally distributed with _13,794_ images per each respective class.

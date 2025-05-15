import csv 
import math 
import random 
import statistics

def cal_probability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent

dataset=[] 
dataset_size=0

with open('lab5.csv') as csvfile:
    lines=csv.reader(csvfile)
    for row in lines:
        dataset.append([float(attr) for attr in row]) 


dataset_size=len(dataset)
print("Size of dataset is: ",dataset_size)
train_size=int(0.7*dataset_size) 
print(train_size)
X_train=[] 
X_test=dataset.copy()

training_indexes=random.sample(range(dataset_size),train_size)
for i in training_indexes: 
    X_train.append(dataset[i]) 
    X_test.remove(dataset[i])

classes={}
for samples in X_train: 
    last=int(samples[-1]) 
    if last not in classes:
        classes[last]=[] 
    classes[last].append(samples)
print(classes) 
summaries={}

for classValue,training_data in classes.items(): 
    summary=[(statistics.mean(attribute),statistics.stdev(attribute)) for attribute in zip(*training_data)] 
    del summary[-1]
    summaries[classValue]=summary
print(summaries) 

X_prediction=[]




for i in X_test: 
    probabilities={}
    for classValue,classSummary in summaries.items(): 
        probabilities[classValue]=1 
        for index,attr in enumerate(classSummary): 
            probabilities[classValue]*=cal_probability(i[index],attr[0],attr[1])
	
    best_label,best_prob=None,-1
    for classValue,probability in probabilities.items(): 
        if best_label is None or probability>best_prob:
            best_prob=probability 
            best_label=classValue 
    X_prediction.append(best_label)

correct=0
for index,key in enumerate(X_test):
   if X_test[index][-1]==X_prediction[index]: 
        correct+=1
print("Accuracy: ",correct/(float(len(X_test)))*100)

#output:
Size of dataset is:  768
537
{0: [(3.324858757062147, 3.052719608069118), (109.48022598870057, 25.283680119833075), (67.48870056497175, 18.67179287851562), (19.5225988700565, 15.055922122555362), (71.7542372881356, 102.84507475449655), (30.280790960451977, 7.90994092128854), (0.42989265536723165, 0.30328868772927187), (30.912429378531073, 11.438997622285232)], 1: [(4.961748633879782, 3.749529084305164), (141.8306010928962, 31.444329721187), (71.08743169398907, 21.135954613726238), (22.060109289617486, 18.291344677138426), (103.44262295081967, 141.9391439698173), (35.320218579234975, 6.641525704853091), (0.5696284153005464, 0.3815471881308416), (37.442622950819676, 11.519965359042374)]}
Accuracy:  70.56277056277057

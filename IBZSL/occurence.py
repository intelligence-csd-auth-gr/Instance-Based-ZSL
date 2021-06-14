import os, pickle

def get_abstract_occurence_preds(labels,abstracts):

    predictions=list()

    for abstract in abstracts:
        flag=True
        string=""

        for label in labels:
            if abstract.__contains__(label) \
                    or abstract.__contains__(label.lower()) \
                    or (label.split(" ")[0] in abstract.split(" ")) \
                    or label.lower().split(" ")[0] in abstract.split(" "):
                flag = False
                if (string != ""):
                    string = string + "#" + label
                elif (string == ""):
                    string = label
        if(flag== True):
            string="None"
        predictions.append(string)

    with open('predictions_label_occurence.pickle', 'wb') as f:
        pickle.dump(predictions, f)
    f.close()
    
    return predictions

#####################################################################################
path = ... #define the path for pre-computed files  
os.chdir(path)

file = open("top_100_labels.txt")
top_labels=list()
for line in file:
    top_labels.append(line[:-1])

test_file = "pure_zero_shot_test_set_top100.txt"
x = list()
y = list()
file = open(test_file)

for line in file:
    y.append(line[2:-2].split("labels: #")[1])
    x.append(line[2:-2].split("labels: #")[0])

preds = get_abstract_occurence_preds(top_labels,x)
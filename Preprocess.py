import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from KnowledgeGraph import KnowledgeGraph
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, hamming_loss
import matplotlib.pyplot as plt

text_tagged_list = []
reason_tagged_list = []

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english')) ## ENGLISH STOP-WORD SET
regex = re.compile(r'[\n\r\t]') ## TO REMOVE UNNECESSARY CHARACTERS LIKE SPACE,TABS,etc

def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 

    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text)

    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
 
    # remove whitespaces 
    text = ' '.join(text.split()) 

    # convert text to lowercase 
    text = text.lower() 
    
    return text

#function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def parseTree(df, column_name):
    tagged_list = []
    allowed_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD']

    for j in df[column_name]:
        tokenized = sent_tokenize(j)
        for i in tokenized:
            wordsList = nltk.word_tokenize(i)
            wordsList = [w for w in wordsList if not w in stop_words]
            tagged = nltk.pos_tag(wordsList)
            tagged = [word for word in tagged if word[1] in allowed_pos]
            tagged_list.append(tagged)
        #print("Tagged list: ", tagged_list)           
    return tagged_list

def simanticSimilarity(text_tagged_list, reason_tagged_list):
    print("Started building knowledge Graph.......")
    graph = KnowledgeGraph(text_tagged_list, reason_tagged_list)
    graph.buildGraph()

    predictions = []
    assert(len(text_tagged_list) == len(reason_tagged_list))
    print("Started calculating semantic from knowledge graph....")
    for phrase in range(len(text_tagged_list)):
        # if(len(reason_tagged_list[phrase]) == 0 or len(text_tagged_list[phrase]) == 0):
        #     continue
        result  = graph.getSemanticSimilary(reason_tagged_list[phrase], text_tagged_list[phrase])
        predictions.append(result)

    return predictions


def predictionMatrice(data_frame, prediction_score_list):
    print("Calculating Prediction Mtrix....")
    data_frame['Prediction'] = ''
    data_frame['Semantic_Similarity_Score'] = ''

    for score in range(len(prediction_score_list)):
        data_frame.loc[score, 'Prediction'] = prediction_score_list[score][1]
        data_frame.loc[score, 'Semantic_Similarity_Score'] = prediction_score_list[score][0]
    
    data_frame['Prediction'] = data_frame['Prediction'].astype(int)
    data_frame['Semantic_Similarity_Score'] = data_frame['Semantic_Similarity_Score'].astype(float)

    count_0 = 0
    count_1 = 0

    for i in data_frame['Prediction']:
        if i == 0:
            count_0 += 1
        else:
            count_1 += 1
    
    print("0: {}, 1: {}".format(count_0, count_1))

    # exp_series = pd.Series(data_frame['label'])
    # pred_series = pd.Series(data_frame['Prediction'])
    # print(pd.crosstab(exp_series, pred_series, rownames=['label'], colnames=['Prediction'],margins=True), "\n")

    #Calculating AUC
    fpr, tpr, threshold = roc_curve( data_frame['label'], data_frame['Prediction'])
    roc_auc = auc(fpr, tpr)
    print("ROC:: fpr: {}, tpr: {}, threshold: {}".format(fpr, tpr, threshold))
    print("AUC:: {}".format(roc_auc), "\n")

    #Calculating Precision, Recall, Accuracy, F1 score, log loss/Binary cross entropy, hamming loss
    print('Positive Precision: %.3f' % precision_score(data_frame['label'], data_frame['Prediction'], pos_label = 1))
    print('Negative Precision: %.3f' % precision_score(data_frame['label'], data_frame['Prediction'], pos_label = 0))
    print('Positive Recall: %.3f' % recall_score(data_frame['label'], data_frame['Prediction'], pos_label = 1))
    print('Negative Recall: %.3f' % recall_score(data_frame['label'], data_frame['Prediction'], pos_label = 0))
    print('Positive F1 Score: %.3f' % f1_score(data_frame['label'], data_frame['Prediction'], pos_label = 1))
    print('Negative F1 Score: %.3f' % f1_score(data_frame['label'], data_frame['Prediction'], pos_label = 0))
    print('Hamming Loss: %.3f' % hamming_loss(data_frame['label'], data_frame['Prediction']))
    print('Accuracy: %.3f' % accuracy_score(data_frame['label'], data_frame['Prediction']))

    #Ploting the Confusion Matrix
    conf_matrix = confusion_matrix(y_true = data_frame['label'] , y_pred = data_frame['Prediction'], labels=[0, 1])
    _ , ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def main(data_frame):
    data_frame['text'] = data_frame['text'].apply(lambda x: clean_text(str(x)))
    data_frame['reason'] = data_frame['reason'].apply(lambda x: clean_text(str(x)))

    data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))
    data_frame['reason'] = data_frame['reason'].apply(lambda x: remove_stopwords(x))


    text_tagged_list = parseTree(data_frame, 'text')
    reason_tagged_list = parseTree(data_frame, 'reason')

    prediction_list = simanticSimilarity(text_tagged_list, reason_tagged_list)

    predictionMatrice(data_frame, prediction_list)

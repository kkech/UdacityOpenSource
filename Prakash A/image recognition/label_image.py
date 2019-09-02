import tensorflow as tf, sys

image_path = input("enter filename: ")
#image_path = greeting
plant = "paddy (oryza sativa)"

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("./retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("./retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

'''#print tensornames
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name, '\n')'''
    
# Feed the image_data as input to the graph and get first prediction
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
        if score >= 0.5:
            print("diagnosis report: ")
            print('%s (score = %.5f)' % (human_string, score))
            report = human_string
            break;

# send results to database
print(report)
# connect to database
import mysql.connector

# connection to mysql database 
myconn = mysql.connector.connect(
  host="localhost",
  database='recommend',
  user="root",
  passwd="Dev12356$"
)

print(myconn)

# select table and execute query 
sql_select_Query = "select * from diag"
cursor = myconn.cursor()
cursor.execute(sql_select_Query)
records = cursor.fetchall()

# selecting row with identified report
sql_Query = "select * from diag where disease = '%s'" % (report)
cursor = myconn.cursor()
cursor.execute(sql_Query)

# fetch single row
record = cursor.fetchone()  

# print report of diagnosis
print("Diagnosis report:", record[1])
print("recommended fertilizer :", record[2])
print("Dosage:", record[3])
report = record[1]
recfert = record[2]
dosage = record[3]

# closing database connection.
if(myconn.is_connected()):
  myconn.close()

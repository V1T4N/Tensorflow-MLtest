
# coding: utf-8

# In[1]:


import keras

(x_train,y_train),(x_test,y_test) = keras.datasets.boston_housing.load_data()


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.hist(y_train,bins=20)
plt.xlabel("house price")
plt.show()

plt.plot(x_train[:,5],y_train,"o",markersize=2)
plt.xlabel("room numbers")
plt.ylabel("price(*$1,000)")
plt.show()


# In[3]:


x_train_mean = x_train.mean(axis = 0) #各要素の平均値 axis = 0:縦方向
x_train_std = x_train.std(axis = 0) #標準偏差

y_train_mean = y_train.mean()
y_train_std = y_train.std()


x_train = (x_train - x_train_mean)/x_train_std
y_train =(y_train - y_train_mean)/y_train_std

x_test = (x_test - x_train_mean)/x_train_std
y_test =(y_test - y_train_mean)/y_train_std

plt.plot(x_train[:,5],y_train,"o",markersize=2)
plt.show()



# In[4]:


import tensorflow as tf

LOG_DIR = "./logs/3_6_3"
# 指定したディレクトリがあれば削除し、再作成
if (tf.gfile.Exists(LOG_DIR)):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)


x = tf.placeholder(tf.float32,(None,13),name = "x")
y = tf.placeholder(tf.float32,(None,1),name = "y")

w = tf.Variable(tf.random_normal((13,1))) #ランダムな値で初期化
pred = tf.matmul(x,w) #予測モデル


# In[5]:


loss = tf.reduce_mean((y - pred)**2) #損失関数の定義　二乗誤差 reduce_meanは平均値を求める関数
tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1) #勾配法で最適化　歩幅=0.1

train_step = optimizer.minimize(loss,var_list = [w]) #wを最適化


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#変数の初期化
    
    
    merged = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    
    
    
    for step in range(100):
        summary,train_loss, _ = sess.run([merged,loss, train_step],feed_dict = {x:x_train, y:y_train.reshape((-1,1))} ) #train_stepの戻り値はNoneなので_に格納
        
        summary_writer.add_summary(summary, step)
    
    pred_ = sess.run(pred,feed_dict ={x:x_test})
    #print(pred_)
        
plt.plot(x_test[:,5],pred_,"o",markersize=2)
plt.xlabel("room")
plt.ylabel("price")
plt.show()
    
summary_writer.close()
    
        


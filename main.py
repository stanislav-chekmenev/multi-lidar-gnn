import bagpy
from bagpy import bagreader

b = bagreader('data/eee_03/eee_03.bag')

# get the list of topics
print(b.topic_table)

# get all the messages of type pointcloud
velmsgs   = b.os1_cloud_node1
veldf = pd.read_csv(velmsgs[0])
plt.plot(veldf['Time'], veldf['linear.x'])
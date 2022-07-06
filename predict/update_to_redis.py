import subprocess
import redis
import argparse
from tqdm import tqdm
class RedisClient:
    def __init__(self, name, pwd=""):
        self.name = name
        self.pwd = pwd
        self.handler = self._get_redis_conn(self.name, pwd)
        self.handler_pipeline = self.handler.pipeline(transaction=False)
    def _get_redis_conn(self, name, pwd=""):
        class ZkConnection(redis.connection.Connection):
            def __init__(self, zkname, **kwargs):
                ip, port = subprocess.Popen(
                    ['zkname', name],
                    stdout=subprocess.PIPE).communicate()[0].split()
                print(ip, port)
                super(ZkConnection, self).__init__(host=ip, port=int(port), **kwargs)

        pool = redis.ConnectionPool(connection_class=ZkConnection, zkname=name, password=pwd)
        rconn = redis.Redis(connection_pool=pool)
        rconn.zkname = name
        return rconn
    def get_keys(self, keys, string_value=False):
        self.handler_pipeline.reset()
        for key in keys:
            self.handler_pipeline.get(key)
        results = self.handler_pipeline.execute()
        results = [x if x is not None else b"" for x in results]
        if string_value:
            results = [x.decode("utf-8") for x in results]
#         print(results)
        self.handler_pipeline.reset()
        return results
    def set_key(self, key, value, expiretime=10 * 86400):
        self.handler_pipeline.reset()
        self.handler_pipeline.setex(key, expiretime, value)
        results = self.handler_pipeline.execute()
    def set_keys(self, keys, values, expiretime=10 * 86400):
        self.handler_pipeline.reset()
        for key, value in zip(keys, values):
            # print("key: {}, value: {}".format(key, value))
            self.handler_pipeline.setex(key, expiretime, value)
        results = self.handler_pipeline.execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", type=str)
    parser.add_argument("--side", default="user", type=str)
    parser.add_argument("--channel", default="bottom", type=str, required=True)
    args = parser.parse_args()
    if args.side == "item":
        # related_news_redis = RedisClient(name="all.news_logoddsratio_item2item.redis.com", pwd="z?GgQy8n/:UY/;Up0")
        related_news_redis = RedisClient(name="all.news_user2user_extend.ssdb.com", pwd="")
    else:
        related_news_redis = RedisClient(name="all.general_recommendation_ucf.redis.com", pwd="4y/wJuvXzHJ7bfHv=nj")
    window = 1000
    keys = []
    values = []
    key_prefix = "{}_cl_i2i#".format(args.channel) if args.side == "item" else "{}_cl_u2u#".format(args.channel)
    print("key_predix: {}".format(key_prefix))
    with open(args.input, "r") as f:
        for line in tqdm(f, desc="update redis results of {}".format(args.input)):
            line = line.strip("\n")
            key, value = line.split("\t")
            key = "{}{}".format(key_prefix, key)
            if len(keys) < window:
                keys.append(key)
                values.append(value)
            else:
                related_news_redis.set_keys(keys=keys, values=values, expiretime=10 * 86400)
                keys = []
                values = []




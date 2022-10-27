#!/usr/bin/env python3
#coding:utf8
import redis

# https://www.cnblogs.com/wang-yc/p/5693288.html

def put():
    r = redis.Redis(host='127.0.0.1', port=6379, db=0,decode_responses=True)
    r.set('foo', 'bar')
    print(r.get('foo'))

if __name__ == '__main__':
    # basic()
    put()
#
from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd
import logging
# Connect to Impala and execute the query

def execute_query(query, cursor=None):
  try:
    impala_con = connect(host='impal-api-internal.hbfintech.com', port=21050, use_ssl=False)
    # If you have a Kerberos auth on your Impala, you could use connection string like:
    # impala_con = connect(host='192.168.250.10', port=21050, use_ssl=True,
    # database='default', user='username', kerberos_service_name='impala',
    # auth_mechanism = 'GSSAPI')
    # NOTE: You may need to install additional OS related packages like:
    # libsasl2-modules-gssapi-mit, thrift_sasl

    impala_cur = impala_con.cursor()
    impala_cur.execute(query)
    result = impala_cur if cursor else impala_cur.fetchall()
    logging.info('Query has been successfully executed')
    impala_cur.close()
    impala_con.close()
    return result
  except Exception as err:
    logging.error('Query execution failed!')
    logging.error(err)
    return None


query = 'select * from dw.dw_product_repay LIMIT 500'
cursor = execute_query(query, cursor=False)
print(cursor)
# data_frame = as_pandas(cursor)
# print(data_frame)


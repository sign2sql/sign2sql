import os
import sqlite3 as sql

in_dbs = ['/mnt/gold/zsj/WikiSQL/data/%s.db' % s 
        for s in ['train', 'dev', 'test']]

out_db = '/mnt/gold/zsj/data/sign2sql/dataset/all.db'

in_tables = ['/mnt/gold/zsj/WikiSQL/data/%s.tables.jsonl' % s 
        for s in ['train', 'dev', 'test']]

out_table = '/mnt/gold/zsj/data/sign2sql/dataset/all.tables.jsonl'

if __name__ == "__main__":
    # merge database
    assert not os.path.exists(out_db)

    cons = sql.connect(out_db)
    for in_db in in_dbs:
        sqls1 = 'attach database "'+in_db+'" as s;'
        cons.execute(sqls1)
        cons.execute("BEGIN")
        for row in cons.execute("SELECT * FROM s.sqlite_master WHERE type='table'"):
            cons.execute(row[-1])  # create table
            combine = "INSERT INTO "+ row[1] + " SELECT * FROM s." + row[1]
            cons.execute(combine)
        cons.commit()
        cons.execute("detach database s")
    cons.close()
    # merge tables.jsonl
    assert not os.path.exists(out_table)
    with open(out_table, 'w') as f:
        for in_table in in_tables:
            with open(in_table) as fin:
                for row in fin.readlines():
                    tmp = row.strip()
                    if len(tmp):
                        f.write(tmp+'\n')


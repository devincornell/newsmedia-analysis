
import semanticanalysis as sa

class Articles(sa.Documents):
    def __init__(self, fname):
        super().__init__(
            fname=fname, 
            tabname='tweets', 
            colschema='id integer primary key autoincrement, source string, date integer, datestr string, url string, meta blob, text string, headline string',
        )
        self.c.execute("create index if not exists idx1 on tweets(date)")
        self.c.execute("create index if not exists idx2 on tweets(source)")
        self.c.execute("create index if not exists idx3 on tweets(date,source)")
        
        
        

.PHONY: all clean distclean

all: train.csv test.csv bids.db

%.csv:
	$(error Missing data file $@)

bids.db: bids.csv
	tail -n +2 $< | sqlite3 $@ ".read bids.sql"

clean:
	rm -f bids.csv

distclean:
	rm -f *.csv *.db

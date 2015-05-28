CREATE TABLE bids(
    "bid_id" INTEGER PRIMARY KEY,
    "bidder_id" TEXT,
    "auction" TEXT,
    "merchandise" TEXT,
    "device" TEXT,
    "time" INTEGER,
    "country" TEXT,
    "ip" TEXT,
    "url" TEXT
);
.separator ","
.mode csv
.import /dev/stdin bids
CREATE INDEX bids_bidder_id ON bids(bidder_id);
CREATE INDEX bids_auction ON bids(auction);

TEXT=~/workspace/oas
DEST=~/workspace/fairseq-oas
mkdir -p $DEST

fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/seq.train.tokens \
    --validpref $TEXT/seq.valid.tokens \
    --testpref $TEXT/seq.test.tokens \
    --destdir $DEST/seq \
    --workers 24

fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/tag.train.tokens \
    --validpref $TEXT/tag.valid.tokens \
    --testpref $TEXT/tag.test.tokens \
    --destdir $DEST/tag \
    --workers 24
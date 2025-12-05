# aws s3 cp --no-sign-request s3://cellpainting-gallery/cpg0020-varchamp/broad/images/2025_05_23_Batch_17/images/2025_04_01_B17A9R1/Images/Index.idx.xml B17.xml

# aws s3 cp --no-sign-request s3://cellpainting-gallery/cpg0020-varchamp/broad/images/2025_05_23_Batch_17/images/2025_04_21_B17A9R2_media_change/Images/Index.idx.xml B17_media_change.xml

tail -n 500 B17_media_change.xml|grep ChannelName|sort -u

tail -n 500 B17.xml|grep ChannelName|sort -u
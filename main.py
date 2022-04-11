# This is a sample Python script.
with open("/home/ps/DiskA/dataset/train1.tsv", "wb") as writer:
    with open("/home/ps/DiskA/dataset/train1.txt", "rb") as f:
        for line in f:
            line = line.decode()
            line = line.split()
            file_name = line[0]
            label = line[1]
            label = label.replace("<space>", " ")
            new_label = ""
            for i in label:
                if i==" ":
                    i = "<space>"
                new_label += i
                new_label += " "
            new_label = new_label.strip()
            print("{}\t{}".format(file_name, new_label))
            writer.write("{}\t{}\n".format(file_name, new_label).encode())
            # break

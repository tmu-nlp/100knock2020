text_1 = "パトカー"
text_2 ="タクシー"
text =""

if len(text_1) != len(text_2):
    print("それぞれの文字数を一致させてください．")
else:
    for i in range (len(text_1)):
        text += text_1[i]
        text += text_2[i]

    print("「" + text_1 + "」と「" + text_2 + "」を交互に連結した文字列は，「" + text + "」です．\n")
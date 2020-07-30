"""
!fairseq-interactive data-bin/kftt.ja-en/ \
    --path checkpoints/kftt.ja-en/checkpoint_best.pt \
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 92.out

!head ../kftt-data-1.0/data/tok/kyoto-test.ja
Infobox Buddhist
道元 ( どうげん ) は 、 鎌倉 時代 初期 の 禅僧 。
曹洞 宗 の 開祖 。
晩年 に 希 玄 と い う 異称 も 用い た 。
同宗旨 で は 高祖 と 尊称 さ れ る 。
諡 は 、 仏性 伝 東 国師 、 承陽 大師 _ ( 僧 ) 。
一般 に は 道元 禅師 と 呼 ば れ る 。
日本 に 歯磨き 洗面 、 食事 の 際 の 作法 や 掃除 の 習慣 を 広め た と い わ れ る 。
最初 に モウ ソウチク ( 孟宗 竹 ) を 持ち帰 っ た と する 説 も あ る 。
道元 の 出生 に は 不明 の 点 が 多 い が 、 内 大臣 土御門 通親 （ 源 通親 あるいは 久我 通親 ） の 嫡流 に 生まれ た と する 点 で は 諸説 が 一致 し て い る 。


!head 92.out
<unk>
Dogen ( <unk> ) was a Zen priest in the early Kamakura period .
He was the founder of the Soto sect .
His name was also used in his later years .
He was also referred to as <unk> .
His posthumous name was <unk> Kokushi ( a priest ) , <unk> ( a priest ) , and <unk> Daishi ( a priest ) .
In general , it is called Dogen Dogen .
It is said that he spread the custom of eating food and customs in Japan .
There is also a theory that he brought back to <unk> <unk> ( <unk> ) .
There are many theories about the birth of Dogen , but there are various theories about the fact that he was born as a direct descendant of Michichika TSUCHIMIKADO ( MINAMOTO no Michichika and Michichika KOGA ) .
"""


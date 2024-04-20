
### Tokenization in LLMS

After exploring character-level transformers, I was inspired by Andrej Karpathy's insights on tokenization, as shared in his [YouTube video](https://www.youtube.com/watch?v=zduSFxRajkE) and his simple byte-pair-encoding (BPE) implementation in the [minbpe repository](https://github.com/karpathy/minbpe/). Motivated by this, I implemented a BasicTokenizer based on the standard BPE algorithm. Additionally, I integrated a RegexTokenizer that, much like the one described in the GPT-2 paper, adds special tokens and segments the text into chunks before applying BPE.

### Project Focus

This repository is dedicated to examining how the size of the vocabulary affects the quality of text generation. We compare character-level language models (with no merges, resulting in a vocabulary size of 256) to models using tokenized text. Tokenizing groups together multiple characters as single tokens, which helps compress the text. This compression not only reduces spelling errors but also allows the model to process longer contexts effectively.

### Training Details

The transformer models were trained on the same dataset consisting of five classic novels (downloaded from Project Gutenberg): 
- Alice's Adventures in Wonderland (Lewis Carroll)
- Around the World in 80 Days (Jules Verne)
- Twenty Thousand Leagues Under the Sea (Jules Verne)
- The Adventures of Tom Sawyer (Mark Twain)
- The Adventures of Sherlock Holmes (Arthur Conan Doyle)

To ensure comparability, each model was trained for approximately 2 hours. The tokenization strategy used groups bytes together and compresses the text, which allows models with larger vocabularies to train on fewer tokens, thereby affording more epochs within the same compute budget.

#### Vocab Size and Epochs

The following table details the relationship between vocabulary size and the number of epochs completed during training:

| Vocab Size | Number of Epochs | Number of Parameters |
| --- | --- | --- |
| 256 | 9 | 5,939,969
| 512 | 18 | 6,071,297
| 1024 | 18 | 6,333,953

#### Model Architecture

The architecture of the transformer was kept consistent across all tokenization strategies. The models were trained using data parallelism on 2 NVIDIA GeForce RTX 3060 GPUs. The specifications of the model architecture are as follows:

| Parameter | Value |
| --- | --- |
| Batch Size | 256 |
| Context Size | 64 |
| Model Dimension (`d_model`) | 256 |
| Number of Heads (`n_heads`) | 8 |
| Number of Layers (`n_layers`) | 8 |

* * *

# Results

| Parameter               | Value |
|-------------------------|-------|
| Sampling Temperature    | 0.8   |
| Number of Samples       | 3     |
| Tokens per Sample       | 128   |

NOTE:

During the development and training of our transformer models, it has been observed that the generated outputs can sometimes include inappropriate/offensive language (e.g. the N-word). This is primarily due to the presence of problematic source material within the training data, notably from the novel "The Adventures of Tom Sawyer."

Prompt:

```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable.
```

vocab size = 256

```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable.  He ran
off from the saloon, saying to himself, “Here are they are! I’m
a-swing on a bend after him.

It was a slight beat d
```
```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable.  He was
very bad luck, and says:

“What’s de use dust one great people terminating up, I could see
them hunt for the raft, s
```
```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable.  The sort of
I went along down the passage of the platform and fell on the water,
or anything in the canoe—you liked the river
```

vocab size: 512

```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable.  His
carried breathing of the bride’s thumb, and, having called Jim
disappeared by degrees; but it was signs of a broken months in it, and
then lightning passed out a few minutes, and the key in which I found
these hands will then carry it out.”

“To the part of the vessel?”


```
```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable.  He
was in the spirit to see if I could but to wait. At last all the
morning at the door, it was obliged to wait. The next morning he
stopped saying “Surely”; and the profession is only just as good
creatures, for the instruments hanging on the train. The long fermented
of co
```
```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable.  That
is, however, the moment it might be troubled, for the commission of the
fishering house was around than one of the business; but, alas, the
captain spoke some company, the reservoirs, encircumbered the years. But if it
was not quite impossible to say whether
```

vocab size: 1024

```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable. 
she was too late to jing it on till her eyes bristling mans to a
kind of whales, but could get a bird; then stuff it in the deep
guard, and called it a solid, and then I couldn’t get it out on the business
and slept badly.

Then the duke says:

“Wouldn’t they give some money, paddled too hopeless, or the latter
for him. I says, I
```
```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable. 
surprise at the end of the source of the news of the landlest, who was
in which moment over landoning the river,
looking at us. At a kind of bed, dagger a small beland, a serpent, with a
_tar_, and a round voice, a spongepor bone, and a table, holding up
the lantern and ronds. The fires had been printed off,
```
```
Alice began to feel very sleepy as she wandered through the garden. Suddenly, she stumbled upon a mysterious object half-buried under a rose bush. It looked old and valuable. 
surprise now, to whom the first ports of the banker’s taste, heat
was betting it and hurled along with horrors and fell laughing. As he writated,
with papers known collaps, and a pocket holding up the shore, and borne a
track on the wall, and the fire and the river softened up, and most
burnt, and everybody just like it wouldn’t
```

Prompt:
```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter.
```

vocab size: 256

```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter.  But
what he did wish to do it just as safe an hour asked me why I was out of
my watch satisfaction. Of these are the bosoms of 
```
```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter.  It
must have been several hours before our night to depart from the
Project Gutenberg™ work (any work
on which you are in con
```
```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter.  Might find
nutting-up, but just shove off a piece of licks and shots, healthy
and some kinds, after all.

They was all over som
```

vocab size: 512

```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter.  Lova
the Spanish coast, and may regard him. Oh, remember his strong week, he
could see himself that he was standing in the last minute. He says:

“If gentlemen and laid of him, and I didn’t want to be hung if I could get
bluying a little, and when I come in the morning all day
```
```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter.  Look
for himself and strongly waiting for him to stop. Had he traversed that
long was made a hansom, driving with my master and saying, “Why, he’s killed
me, s’I.”

His says:

“Dah you goes, de ole true Huck, they don’t get away from here?”

Then he was after it, and says
```
```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter.  Low
the little girls in like a skiff, and she done back to the middle of the
cup of tea.

“I’ll just have every day or two to pray without being able to tie to any
of these snow.”

“Let me see,” said Holmes, “I only walked round it into the midst of the
roaring of the broad brilliant
```

vocab size: 1024

```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter. 
now as if it were all arranged; the attitude was anxious to shut the
strange affair; but not better than better by the same monster: but it
was drunk, I should never have easy and common so. Pretty soon I made up my
hand, and says:

“Say, look at that, now!” says the king, kinder gang was the big
fire to one side. But the boy was full of gun and
```
```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter. 
three-six in the morning, it was clearly drink, but it was now only
beginning to wait for the house.”

Our visitor staggered to his master what he had seated himself. I was
rather of an instant; but he had noticed some man, though he had not
made his hastening between his lips. Aouda, wandering holdly at the
depfortunate man, with the silence of the
```
```
Phileas Fogg had just returned to London, believing he had lost his wager, when he received a strange letter. 
surprised at the very highly Indian cigar, which might have been
alarmed by a very 4th which he was stated by the twenty-five miles an
hour, and two miles below New Orleans; and this were the forest, if Fix
had remained to take the sea use of him. Passepartout, who was once
more in the neighbourhood of the junction of his shiny, threw them in
him
```

Prompt:
```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing.
```

vocab size: 256

```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing. 

“I’ve been in that; and I’ll answer you to do what you can?”

“I shall only use my chronometer.”

“What could yo
```
```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing. ”

“They are going, then, for at full state that I was reasonable. I thought
that Captain Nemo was there.

“Many of the An
```
```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing. ”

“I assure you, that you have heard pretty far in town, and mind to rip
it off. So I says:

“But the brute _Henrietta”
```

vocab size: 512

```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing.  I
could see that the maiden herself was so slight that her very few of the
young woman. Aronnax, of course, of the people, of profound
terradiation, and these electric light which flooded us, I could
distinguish the deepest thought, with the interest in her life rocks, when she
went
```
```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing.  I
was offered to say that the danger would be as pressing in one house as in
another.

“It would be something _on_.”

Aouda’s maiden slow, and then in the peace of a very dogs; so I knowed
it wouldn’t hardly any more, or else he wouldn’t. Said he jis’
wished he didn’t get drunk 
```
```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing.  I
expected myself, for the only process of having visited the advertisement,
appeared the Canadian informed him of the missing gaudily chance has
been in the hammer-house.”

So him, but he went on, calling after himself a terrible strange exist—ive
shaking hand, there has be
```

vocab size: 1024

```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing. 
she heard the floor of the wigwam. Jim was very deep, and was going to
tell Mr. Fogg’s teeth again, and notice-rately laughed; and, as he
found himself, looking at the middle of the dark, which was nearly
easily accustomed. He sawres of having only taken a kniang, which is
also five yards inches long, the millions were filled with sands
```
```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing. 
now ten yards broaded, I sent down a pathway through the Captain’s
singular adventures which were furious. It was not a the first different
minutes, but, chet for being solved, with bracelets, with gigantic
between the brick factories and which, inspectors, if you will leave
our honour to your example.”

I had taken my way into a p
```
```
In the quiet waters of the Mississippi, Huck Finn spotted something unusual floating towards the raft one foggy evening. As it got closer, he realized it was not debris, but something much more intriguing. 
three-five thousand-twenty-five miles, and a look, and the devil
water was laid up in a
strong shot. I looked at the shore, and heard of seeing it opening again,
and I waited to find out whether the canoe could let me down just as it
was nearly dark. They warn’t ever so much. Not a man that was
dreadful. There warn’t any but one idea, and the candle is
```

Prompt:
```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection.
```

vocab size: 256

```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection.  The
whole sheet was suppressed. The _Nautilus_ had stopped. I admired the curious
crew, but, I should see him no more. I see Ho
```
```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection. ”

“I am not,” said the March Hare.

“For two who are here?” I asked.

“I never will soon smell before he got a chan
```
```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection.  It
glanced at the bottom of the door and stretch a line of flag to meet, so that it
will be the best hours against the wall. Th
```

vocab size: 512

```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection.  The
director certainly mean the thing I would come. I shall want you to
water it with your tears.”

I could see that I. I have my chair trunks (which is my master; I can’t
tell you it about it.”

“No,” says the old man, “I reckon we might say a word.”

I hope so. But pap on the table spoke out of
```
```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection.  During
the marvels of the ocean those zoophytes pressed around so dear little.
You couldn’t have come but sure they had all the trouble in the world,
that navigators. They were very anxious to marry among the numerous
judgment was soon had calculated. On the 15th of March we
```
```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection.  During
the presses of the water was at once drawn up to the surface of
the sea, a struggle between the two verdant date. A vast country pursue Project
Gutenberg™ electronic works in your possession.” And Ned Land
Aouda, without leaving the
```

vocab size: 1024

```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection. 
must speaking, shut the breeze, swoided its silence, gave the
expression became more calm, and which could be immovable. I
repeatedly the bearings of a brave chance, the _Nautilus_ floated in a
lantern. It was a deposition of about twelve miles an hour, but its
disast eggs appeared on the shore, and I found a map of de
```
```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection. 
Captain Nemo, whose brightened, projected grey eyes. Ruged at the
purpose of the waves, was the breeze, the breeze, and the brewer
guide, the magnificent almost southward. Besides, it is just
possible that the monster did not seem to Yokohama, and being carried on
having a detail of day; and on fifteen minutes left
```
```
Sherlock Holmes received a cryptic note that read simply, 'The giant squid awaits.' Knowing Holmes's penchant for puzzles, this message plunges him into a deep reflection. 
now-swum house in a mouse, of a quiet lady could not happen upon the
first, and yet it was all paid for him to be ready for the widow’s
fooling-truck. In the only stretch then up to the door, and then shook me
herself in the midst of a basket, and dressed him out towards the
shelf business. I set there and put it all still. I couldn’t wait a
```

Prompt:
```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land.
```

vocab size: 256

```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land.  So the
old little thing go about her machinery in its large forests, that were massed
together round the house, and seemed all 
```
```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land.  This soon found
the dead made the calf and let Sid down the river any little right
money and never see anybody like it was a li
```
```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land. 

I saw the second lieutenant light, during the squalls of the carbonic acid with
surprise. “Watson, like a common servant giv
```

vocab size: 512

```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land.  At
two cable last he remained when I saw the centre of the cards I had
noished by the side of the cars, but the two friends will not allow
without its disposal, or common cachalot contained, however, that
they have had been drawn out by two hours we have reached the
station. Medicine
```
```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land.  We
me soon relict themselves to find the south the mountain was a vast
signal of London. Mr. Fogg might live on a double line of things,
    “That is not the chief of the cash is not really enough. But I
have stole Cusan still I was busy, and we shall just wait for
```
```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land.  At
two careless than thirty miles from the _Nautilus_.

For a whole hour was I deep in these reflections, seeking to gain a
sort of pulling open out a carriage in the woods, and catching the
players, extinguished the immense darkness.

I climbed the mizzen-m
```

vocab size: 1024

```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land. 
now, when there was one of the scent occurred which I was keeping on earth
and open, and I could see nothing but a strange dress which I took my bank
to easy open to me, and I found my smack ones and stopped at the dog.
There was a perfect rifle of cards, around the parlor, and out-of-wheel
whispered, and big old snakes, and spiders, and a
```
```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land. 
Captain Nemo, dancing hotels of men on the pillows. He looked
slow, and was an old scrape, and skilk upon the floor door of the eye
two rose at and out for a match. I passed through the danger which was
burnt. The place it began to hold the branches of the large snories, but
concluded indicated anger at the end of the detail
```
```
While exploring the depths of the ocean in the Nautilus, Captain Nemo discovers an underwater cave filled with artifacts. Among these, he finds a detailed map that leads to a hidden location on land. 
Certainly, in the midst of a woman’s apronicle’s name. “I do not fish for
me, for I have one or two pounds.”

“But, sir, I think that the world has been finished.”

“And now?” I asked, contracting it fully with a frightened eyes. “In that
is—”

“Very well, sir.”

“But what do you say, Professor?”

“Why not?”

“You must not be solved.”

“How
```
import predict


def test_predict():
    review_good = "this film was just the brilliant casting location scenery story direction everyone's really suited the " \
             "part they played and you could just imagine being there robert <UNK> is an amazing actor and now the same " \
             "being director <UNK> father came from the same scottish island as myself so i loved the fact there was a real " \
             "connection with this film the witty remarks throughout the film were great it was just brilliant so much that " \
             "i bought the film as soon as it was released for <UNK> and would recommend it to everyone to watch and the " \
             "fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film " \
             "it must have been good and this definitely was also <UNK> to the two little boy's that played the <UNK> of " \
             "norman and paul they were just brilliant children are often left out of the <UNK> list i think because the " \
             "stars that play them all grown up are such a big profile for the whole film but these children are amazing " \
             "and should be praised for what they have done don't you think the whole story was so lovely because it was " \
             "true and was someone's life after all that was shared with us all"

    review_bad = "big hair big boobs bad music and a giant safety pin these are the words to best describe this terrible movie " \
       "i love cheesy horror movies and i've seen hundreds but this had got to be on of the worst ever made the plot " \
       "is paper thin and ridiculous the acting is an abomination the script is completely laughable the best is the " \
       "end showdown with the cop and how he worked out who the killer is it's just so damn terribly written the " \
       "clothes are sickening and funny in equal <UNK> the hair is big lots of boobs <UNK> men wear those cut <UNK> " \
       "shirts that show off their <UNK> sickening that men actually wore them and the music is just <UNK> trash that " \
       "plays over and over again in almost every scene there is trashy music boobs and <UNK> taking away bodies and " \
       "the gym still doesn't close for <UNK> all joking aside this is a truly bad film whose only charm is to look " \
       "back on the disaster that was the 80's and have a good old laugh at how bad everything was back then "

    result_good, classification_good = predict.predict(review_good)
    assert result_good > 0.5
    assert classification_good == "good"

    result_bad, classification_bad = predict.predict(review_bad)
    assert result_bad < 0.5
    assert classification_bad == "bad"

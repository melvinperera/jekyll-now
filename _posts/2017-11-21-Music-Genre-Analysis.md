---
layout: post
title: Music Genre Analysis
---

In this project, I explored the ability of convolutional neural networks to extract features from spectrograms of songs. A model was trained on these features to perform classification of their respective genres. 

## Project Motivation

The feeling that we get when we listen to music - the very sound waves that vibrate our ear drums and fire up our brains in response - is as subjective as it gets. There are no two ways about it: one person might enjoy the soft and ethereal tunes of dream pop, while another would enjoy the intensity and power of industrial metal. Generally, _people tend to stick with and listen to just a few musical genres that they favour or have an affinity with_, be it pop, rock, jazz, or classical music, to name a few. 

What defines a music genre? Does the answer lie in the instruments that play in the song? Is it defined by the speed in which the music plays? Or is it the message that the singer is trying to get across? Maybe, it is a combination of all of those elements, underpinned by an unspoken rule that permeates through everybody's consciousness? Without any information - the title of the song, the artist's name, the date the song was released - a person would easily be able to identify the genre of a song in a few seconds to a high degree of accuracy.

This brings me to my project. __What I am trying to do in this project is to develop a model to perform accurate classification of music genres to a high degree of accuracy. The only data that will be available to the model is the raw audio file itself - no metadata or other information will be given other than the actual audio for classification.__

#### Why is this important?

Parties that would benefit most from robust models to extract audio information is are music companies like Spotify, SoundCloud, or Apple. For example, Spotify receives [thousands](https://expandedramblings.com/index.php/spotify-statistics/) of new tracks a day, and having a model to extract audio features from the tracks would be invaluable for classification. In particular, their playlist generation and music recommendation system benefits from augmenting a traditional [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) based recommender with content-based recommenders to aid with the [cold start problem](https://en.wikipedia.org/wiki/Cold_start) for new tracks that are recently uploaded. 

Aside from the music industry, there could be significant benefits to be had with other industries. With this, it is possible that any audio could potentially be run through a similar model, and important features could be extracted and analysed or classified.

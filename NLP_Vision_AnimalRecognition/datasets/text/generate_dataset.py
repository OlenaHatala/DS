import json
import random
import os

animal_classes = ["dog", "cat", "chicken", "horse", "butterfly", 
           "elephant", "squirrel", "spider", "sheep", "cow", 
           'alligator', 'bear', 'fox', 'wolf', 'pig']

sentence_templates = [
    "I saw a {animal} in the park.",
    "The {animal} is running fast.",
    "A {animal} is a great pet.",
    "There is a {animal} on the farm.",
    "The {animal} loves to play outside.",
    "I don't know where the {animal} is.",
    "The {animal} is very friendly.",
    "Look at that {animal} in the zoo!",
    "I wonder how the {animal} is doing.",
    "I heard a {animal} is good at hunting.",
    "It is not easy to catch a {animal}.",
    "A {animal} can be dangerous sometimes.",
    "I don't like to be around a {animal}.",
    "The {animal} is so cute and fluffy.",
    "There was a {animal} near the river.",
    "A {animal} likes to swim in the pond.",
    "Have you seen the {animal} today?",
    "The {animal} is sleeping in the barn.",
    "I think the {animal} is hungry.",
    "The {animal} just chased a butterfly.",
    "I took a picture of the {animal} in the field.",
    "The {animal} is playing with a ball.",
    "Did you hear that? It was a {animal}.",
    "The {animal} loves to climb trees.",
    "A {animal} can be very loyal.",
    "The {animal} is so graceful when it moves.",
    "I saw a {animal} at the animal shelter.",
    "A {animal} would be a perfect companion.",
    "It is hard to find a {animal} in the wild.",
    "I think the {animal} is lost.",
    "The {animal} enjoys taking long naps.",
    "I wonder if the {animal} can fly.",
    "The {animal} just made a funny sound.",
    "The {animal} is really good at hiding.",
    "A {animal} was playing in the garden today.",
    "We spotted a {animal} near the lake.",
    "I love how the {animal} sounds in the morning.",
    "The {animal} is a symbol of good luck.",
    "I never knew a {animal} could do that!",
    "Look, there’s a {animal} chasing its tail!",
    "The {animal} seems to be very intelligent.",
    "I saw a {animal} running across the field.",
    "The {animal} is eating some food.",
    "Did you see the {animal} in the tree?",
    "A {animal} is watching us from afar.",
    "The {animal} just jumped over the fence.",
    "I can hear the {animal} calling out.",
    "The {animal} is looking for food.",
    "The {animal} is trying to catch a fish.",
    "I can’t believe the {animal} did that!",
    "I saw the {animal} chasing something.",
    "The {animal} has an interesting color.",
    "A {animal} is always curious about new things.",
    "The {animal} looked at me with big eyes.",
    "I watched the {animal} play with a ball.",
    "A {animal} might get scared easily.",
    "The {animal} made a funny noise.",
    "The {animal} is walking around the yard.",
    "I can't wait to see the {animal} again.",
    "The {animal} is sitting quietly in the corner.",
    "A {animal} likes to dig holes in the ground.",
    "The {animal} enjoys sunbathing in the garden.",
    "The {animal} made its home in a tree.",
    "I saw a {animal} flying high in the sky.",
    "The {animal} loves to chase after toys.",
    "The {animal} likes to stay in the shade.",
    "A {animal} is often playful and energetic.",
    "The {animal} is taking a nap in the sun.",
    "I wish I had a {animal} to play with.",
    "The {animal} has very sharp claws.",
    "The {animal} is playing with a stick.",
    "I found a {animal} near the barn.",
    "The {animal} walked slowly towards me.",
    "A {animal} sometimes likes to hide in tight places.",
    "The {animal} is eating some grass.",
    "The {animal} loves to jump around.",
    "The {animal} is looking for its family.",
    "There is a {animal} in the yard right now.",
    "A {animal} often makes a lot of noise.",
    "The {animal} keeps running in circles.",
    "Look at how fast the {animal} can run!",
    "The {animal} is playing with another {animal}.",
    "A {animal} can make a lot of noise.",
    "The {animal} jumped into the water."
]

def generate_ner_dataset(num=3000):
    dataset = []

    for _ in range(num):
        animal = random.choice(animal_classes)

        sentence = random.choice(sentence_templates).format(animal=animal)

        tokens = sentence.split()

        labels = ["O"] * len(tokens)

        for i, token in enumerate(tokens):
            if token.lower() == animal:
                labels[i] = f"B-{animal.upper()}"

        dataset.append({"tokens": tokens, "labels": labels})

    return dataset


def save_dataset(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(dataset, f, indent=4)


# if __name__ == "__main__":
#     dataset = generate_ner_dataset(3000)
#     print(dataset[:5])

#     save_dataset(dataset, "./ner_dataset.json")


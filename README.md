# Latent-Representation-Variational-AutoEncoder
A fun representation of the latent space

Playing around with latent 10-digit scribble representations using the well-known deep learning Variational AutoEncoders architecture.

Nothing new under the sun, known architecture and a project more than known. A brief explanation of what magic I have done here:
Imagine you have a paragraph to memorize for an exam for tomorrow. Obviously you do not see the usefulness of learning it word for word as a hard drive and instead you get two or three fundamental ideas that, you are sure, will allow you in the exam to write a paragraph of similar length but with slightly different words.
This is exactly what the family of neural networks called AutoEncoders do, they create a representation by taking out the most important "ideas" (characteristics, styles, ...) of a large number of numbers (in this case 2D points) and then they can recreate them from what that you have learned.

The "bottleneck" architecture has thousands of uses in Deep Learning, but one that gives food for thought would be to imagine being able to store images, texts or videos in latent spaces of a few coordinates and then remove them almost without loss. That a 1Gb video can be compressed to a point in a coordinate system.

Video uploaded on my Linkedln: https://www.linkedin.com/feed/update/urn:li:ugcPost:6671820044458057728/

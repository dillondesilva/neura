# Tools within this suite

Neura's Git repo is essentially split into three critical components:

- **neura-app**: This is the actual application side of Neura (ie: What our users will see)
- **buddy**: This is a production grade machine learning utility that will help power Neura's immersive gesture recognition capabilities
- **R&D**: This is our prototyping work and is generally built with an end user approach to validate an idea or test a mechanism

## Using buddy

Buddy's core functionality lies in performing two main operations:

- Profiling hand image data through MediaPipe to quickly and efficiently produce training data

- Allowing developers to easily create models without having to write any code

# How to run

Once you are in the ```neura-app``` directory, simply run 

```
npm i -g serve
serve
```
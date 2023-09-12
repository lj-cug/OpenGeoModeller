# JAVA中的CPU耗时计算

https://www.baeldung.com/java-measure-elapsed-time

## 1. Overview

In this article, we're going to have a look at how to measure elapsed time in Java. While this may sound easy, there're a few pitfalls that we must be aware of.

We'll explore standard Java classes and external packages that provide functionality to measure elapsed time.

## 2. Simple Measurements

### 2.1. currentTimeMillis()

When we encounter a requirement to measure elapsed time in Java, we may try to do it like:

long start = System.currentTimeMillis();
// ...
long finish = System.currentTimeMillis();
long timeElapsed = finish - start;

If we look at the code it makes perfect sense. We get a timestamp at the start and we get another timestamp when the code finished. Time elapsed is the difference between these two values.

However, the result may and will be inaccurate as System.currentTimeMillis() measures wall-clock time. Wall-clock time may change for many reasons, e.g. changing the system time can affect the results or a leap second will disrupt the result.


### 2.2. nanoTime()

Another method in java.lang.System class is nanoTime(). If we look at the Java documentation, we'll find the following statement:

“This method can only be used to measure elapsed time and is not related to any other notion of system or wall-clock time.”

Let's use it:

long start = System.nanoTime();
// ...
long finish = System.nanoTime();
long timeElapsed = finish - start;

The code is basically the same as before. The only difference is the method used to get timestamps – nanoTime() instead of currentTimeMillis().

Let's also note that nanoTime(), obviously, returns time in nanoseconds. Therefore, if the elapsed time is measured in a different time unit we must convert it accordingly.

For example, to convert to milliseconds we must divide the result in nanoseconds by 1.000.000.

Another pitfall with nanoTime() is that even though it provides nanosecond precision, it doesn't guarantee nanosecond resolution (i.e. how often the value is updated).
However, it does guarantee that the resolution will be at least as good as that of currentTimeMillis().

## 3. Java 8

If we're using Java 8 – we can try the new java.time.Instant and java.time.Duration classes. Both are immutable, thread-safe and use their own time-scale, the Java Time-Scale, as do all classes within the new java.time API.

### 3.1. Java Time-Scale
The traditional way of measuring time is to divide a day into 24 hours of 60 minutes of 60 seconds, which gives 86.400 seconds a day. However, solar days are not always equally long.

UTC time-scale actually allows a day to have 86.399 or 86.401 SI seconds. An SI second is a scientific “Standard International second” and is defined by periods of radiation of the cesium 133 atom). This is required to keep the day aligned with the Sun.

The Java Time-Scale divides each calendar day into exactly 86.400 subdivisions, known as seconds. There are no leap seconds.

### 3.2. Instant Class

The Instant class represents an instant on the timeline. Basically, it is a numeric timestamp since the standard Java epoch of 1970-01-01T00:00:00Z.

In order to get the current timestamp, we can use the Instant.now() static method. This method allows passing in an optional Clock parameter. If omitted, it uses the system clock in the default time zone.

We can store start and finish times in two variables, as in previous examples. Next, we can calculate time elapsed between both instants.

We can additionally use the Duration class and it's between() method to obtain the duration between two Instant objects. Finally, we need to convert Duration to milliseconds:

Instant start = Instant.now();
// CODE HERE        
Instant finish = Instant.now();
long timeElapsed = Duration.between(start, finish).toMillis();

## 4. StopWatch

Moving on to libraries, Apache Commons Lang provides the StopWatch class that can be used to measure elapsed time.

### 4.1. Maven Dependency

We can get the latest version by updating the pom.xml:
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-lang3</artifactId>
    <version>3.12.0</version>
</dependency>

The latest version of the dependency can be checked here.

### 4.2. Measuring Elapsed Time With StopWatch

First of all, we need to get an instance of the class and then we can simply measure the elapsed time:

StopWatch watch = new StopWatch();
watch.start();

Once we have a watch running, we can execute the code we want to benchmark and then at the end, we simply call the stop() method. Finally, to get the actual result, we call getTime():

watch.stop();
System.out.println("Time Elapsed: " + watch.getTime()); // Prints: Time Elapsed: 2501

StopWatch has a few additional helper methods that we can use in order to pause or resume our measurement. This may be helpful if we need to make our benchmark more complex.

Finally, let's note that the class is not thread-safe.

## 5. Conclusion

There are many ways to measure time in Java. We've covered a very “traditional” (and inaccurate) way by using currentTimeMillis(). Additionally, we checked Apache Common's StopWatch and looked at the new classes available in Java 8.

Overall, for simple and correct measurements of the time elapsed, the nanoTime() method is sufficient. It is also shorter to type than currentTimeMillis().

Let's note, however, that for proper benchmarking, instead of measuring time manually, we can use a framework like the Java Microbenchmark Harness (JMH). This topic goes beyond the scope of this article but we explored it here.

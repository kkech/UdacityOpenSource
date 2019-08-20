object Versions {
    // App versioning
    const val appId = "com.mccorby.openmined.worker"
    const val appVersionCode = 1
    const val appVersionName = "1.0"

    const val compileSdk = 28
    const val minSdk = 26
    const val targetSdk = 28

    const val gradle = "3.3.1"
    const val kotlin = "1.3.31"
    const val coroutines = "1.1.1"
    const val buildTools = "28.0.3"

    // Android libraries
    const val appCompat = "28.0.0"
    const val multiDex = "1.0.3"
    const val workManager = "1.0.0-alpha10"

    // Arch components
    const val archComponents = "1.1.1"

    // Arrow
    const val arrowVersion = "0.8.2"

    // DL4J
    const val dl4j = "1.0.0-beta4"
    const val openblas = "0.3.5-1.5"
    const val opencv = "4.0.1-1.5"
    const val leptonica = "1.78.0-1.5"
    const val gson = "2.8.2"
    const val lombok = "1.16.16"

    // Data source
    const val socketIO = "1.0.0"
    const val msgpack = "0.8.16"
    const val lz4 = "1.4.0"
    const val okhttp = "3.14.0"

    // Tools
    const val ktlint = "0.32.0"
    const val npy = "0.3.3"
    const val rxJava = "2.2.7"
    const val rxAndroid = "2.1.1"

    // Test
    const val junit = "4.12"
    const val testRunner = "1.0.2"
    const val espresso = "3.0.2"
    const val mockk = "1.9.3"
}

object ProjectDependencies {
    const val androidGradlePlugin = "com.android.tools.build:gradle:${Versions.gradle}"
    const val kotlinGradlePlugin = "org.jetbrains.kotlin:kotlin-gradle-plugin:${Versions.kotlin}"
}

object MainApplicationDependencies {
    const val kotlin = "org.jetbrains.kotlin:kotlin-stdlib-jdk7:${Versions.kotlin}"
    const val coroutines = "org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}"
    const val appCompat = "com.android.support:appcompat-v7:${Versions.appCompat}"
    const val multiDex = "com.android.support:multidex:${Versions.multiDex}"
    const val rxJava = "io.reactivex.rxjava2:rxjava:${Versions.rxJava}"
    const val rxAndroid = "io.reactivex.rxjava2:rxandroid:${Versions.rxAndroid}"
    const val archComponentLifeCycleExtensions = "android.arch.lifecycle:extensions:${Versions.archComponents}"
    const val archComponentLifeCycleViewModel = "android.arch.lifecycle:viewmodel:${Versions.archComponents}"
    const val workerManager = "android.arch.work:work-runtime-ktx:${Versions.workManager}"
}

object DL4JDependencies {
    const val dl4j = "org.deeplearning4j:deeplearning4j-core:${Versions.dl4j}"
    const val nd4jNative = "org.nd4j:nd4j-native:${Versions.dl4j}"
    const val nd4jNativeArm = "org.nd4j:nd4j-native:${Versions.dl4j}:android-arm"
    const val nd4jNativeArm64 = "org.nd4j:nd4j-native:${Versions.dl4j}:android-arm64"
    const val nd4jNativeX86 = "org.nd4j:nd4j-native:${Versions.dl4j}:android-x86"
    const val nd4jNativeX86_64 = "org.nd4j:nd4j-native:${Versions.dl4j}:android-x86_64"

    const val openblas = "org.bytedeco:openblas:${Versions.openblas}"
    const val openblasAndroidArm = "org.bytedeco:openblas:${Versions.openblas}:android-arm"
    const val openblasAndroidArm64 = "org.bytedeco:openblas:${Versions.openblas}:android-arm64"
    const val openblasAndroidX86 = "org.bytedeco:openblas:${Versions.openblas}:android-x86"
    const val openblasAndroidX86_64 = "org.bytedeco:openblas:${Versions.openblas}:android-x86_64"

    const val opencv = "org.bytedeco:opencv:${Versions.opencv}"
    const val opencvAndroidArm = "org.bytedeco:opencv:${Versions.opencv}:android-arm"
    const val opencvAndroidArm64 = "org.bytedeco:opencv:${Versions.opencv}:android-arm64"
    const val opencvAndroidX86 = "org.bytedeco:opencv:${Versions.opencv}:android-x86"
    const val opencvAndroidX86_64 = "org.bytedeco:opencv:${Versions.opencv}:android-x86_64"

    const val leptonica = "org.bytedeco:leptonica:${Versions.leptonica}"
    const val leptonicaAndroidArm = "org.bytedeco:leptonica:${Versions.leptonica}:android-arm"
    const val leptonicaAndroidArm64 = "org.bytedeco:leptonica:${Versions.leptonica}:android-arm64"
    const val leptonicaAndroidX86 = "org.bytedeco:leptonica:${Versions.leptonica}:android-x86"
    const val leptonicaAndroidX86_64 = "org.bytedeco:leptonica:${Versions.leptonica}:android-x86_64"

    const val gson = "com.google.code.gson:gson:${Versions.gson}"
    const val lombok = "org.projectlombok:lombok:${Versions.lombok}"
}

object DataSourceDependencies {
    const val socketIO = "io.socket:socket.io-client:${Versions.socketIO}"
    const val msgpack = "org.msgpack:msgpack-core:${Versions.msgpack}"
    const val lz4 = "org.lz4:lz4-java:${Versions.lz4}"
    const val okhttp = "com.squareup.okhttp3:okhttp:${Versions.okhttp}"
}

object ToolsDependencies {
    const val ktlint = "com.pinterest:ktlint:${Versions.ktlint}"
    const val npy = "org.jetbrains.bio:npy:${Versions.npy}"
}

object UnitTestDependencies {
    const val junit = "junit:junit:${Versions.junit}"
    const val mockk = "io.mockk:mockk:${Versions.mockk}"
}

object InstrumentationDependencies {
    const val testRunner = "com.android.support.test:runner:${Versions.testRunner}"
    const val espressoCore = "com.android.support.test.espresso:espresso-core:${Versions.espresso}"
}

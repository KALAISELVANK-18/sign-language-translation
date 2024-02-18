//Package imports
import 'package:flutter/material.dart';

//File imports
import 'package:meet/screens/home_screen.dart';
import 'package:meet/speech.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;



void main() {

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Flutter Meet',
      theme: ThemeData(
          primaryColor: Colors.grey[900],
          outlinedButtonTheme: OutlinedButtonThemeData(
              style: ButtonStyle(
                  shape: MaterialStateProperty.all(RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20.0))),
                  backgroundColor: MaterialStateColor.resolveWith(
                      (states) => Colors.blueAccent),
                  foregroundColor: MaterialStateColor.resolveWith(
                      (states) => Colors.white)))),
      home: SpeechToTextScreen(),
    );
  }
}

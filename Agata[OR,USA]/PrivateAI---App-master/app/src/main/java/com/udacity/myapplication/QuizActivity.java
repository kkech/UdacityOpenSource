package com.udacity.myapplication;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import java.util.List;

public class QuizActivity extends AppCompatActivity {
    private TextView textViewQuestion;
    private TextView textViewScore;
    private TextView textViewQuestionCount;
    private TextView textViewCountDown;
    private RadioGroup rbGroup;
    private RadioButton rb1;
    private RadioButton rb2;
    private RadioButton rb3;
    private RadioButton rb4;
    private Button buttonConfirmNext;
    private List<Question> questionList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_quiz);

//        textViewQuestion = findViewById(R.id.text_view_question);
//        textViewScore = findViewById(R.id.text_view_score);
//        textViewQuestionCount = findViewById(R.id.text_view_question_count);
//        textViewCountDown = findViewById(R.id.text_view_countdown);
//        rbGroup = findViewById(R.id.radio_group);
//        rb1 = findViewById(R.id.radio_button1);
//        rb2 = findViewById(R.id.radio_button2);
//        rb3 = findViewById(R.id.radio_button3);
//        rb4 = findViewById(R.id.radio_button4);
//        buttonConfirmNext = findViewById(R.id.button_confirm_next);
//
//        QuizDBHelper dbHelper = new QuizDBHelper(this);
//        questionList = dbHelper.getAllQuestions();
    }
}

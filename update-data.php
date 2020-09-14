<?php 
session_start();

include 'co1.php';


if($_SERVER['REQUEST_METHOD'] == 'POST'){
   unset($_SESSION['errors']);

    $psw1 = filter_var($_POST['psw1'], FILTER_SANITIZE_STRING);
    $psw2= filter_var($_POST['psw2'], FILTER_SANITIZE_STRING); 
    $tel=filter_var($_POST['tel'], FILTER_SANITIZE_STRING); 

      $error_msg = [];
       if($psw1 == ''||$psw2== ''){
        $error_msg[] = 'password is required';
       }
    
       if (strlen($psw1) <= '8') {
            $error_msg[] = "Your Password Must Contain At Least 8 Characters!";
        }
        elseif(!preg_match("#[A-Z]+#",$psw1)) {
            $error_msg[] = "Your Password Must Contain At Least 1 Capital Letter!";
        }
        elseif(!preg_match("#[a-z]+#",$psw1)) {
            $error_msg[] = "Your Password Must Contain At Least 1 Lowercase Letter!";
        } 
        elseif(!preg_match('/[\'^£$%&*()}{@#~?><>,|=_+¬-]/', $psw1)) {
        $error_msg[]= "Your Password Must Contain At Least 1 Special Character !"."<br>";
         }
        elseif(!preg_match("#[0-9]+#",$psw1)){
         $error_msg[] = "Your Password Must Contain At Least 1 Number!";
        }
        else {
        }
   
    if($psw1!=$psw2){

        $error_msg[] = 'confirm password and the password must be the same';

    }
      $data = [
        'psw1' => $psw1,
        'psw2' => $psw2,
        'tel' => $tel
    ];
    if(empty($error_msg)){
      $sql=' update company set password=:psw1 where phone=:tel';
        $stmt=$conn->prepare($sql);
      $stmt->execute(array('psw1' => $psw1,'tel' => $tel));
       header('Location:login.php');
     
     exit();
    
    }
    else{

        $_SESSION['errors'] = $error_msg;
        $_SESSION['data'] = $data;
        header('Location:update-password.php');
        exit();
    }







    }
 ?>

<?php 
session_start();

include 'co1.php';

if($_SERVER['REQUEST_METHOD'] == 'POST'){
   unset($_SESSION['errors']);
   unset($_SESSION['data']);

    $companyname= filter_var($_POST['companyname'], FILTER_SANITIZE_STRING);
    $email= filter_var($_POST['email'], FILTER_SANITIZE_STRING);
    $phone= filter_var($_POST['phone'], FILTER_SANITIZE_STRING);
    $city = filter_var($_POST['city'], FILTER_SANITIZE_STRING);
    $area= filter_var($_POST['area'], FILTER_SANITIZE_STRING);
    $psw1 = filter_var($_POST['psw1'], FILTER_SANITIZE_STRING);
    $psw2= filter_var($_POST['psw2'], FILTER_SANITIZE_STRING); 
    //echo $companyname.$email.$phone.$city.$area.$psw1.$psw2;
      $error_msg = [];

    if(empty($companyname)){
        $error_msg[] = 'company name is required';
    }
     if(!filter_var($email, FILTER_VALIDATE_EMAIL)){
        $error_msg[] = 'Enter valid email';
    }
    if($phone!=''){
        if(strlen($phone) != 11 && !preg_match('/01[0125]{1}[0-9]{8}/', $phone ))
            $error_msg[] = 'Enter valid phone number';
    }
    if($city== ''){
        $error_msg[] = 'city name is required';
    }
    if($area == ''){
        $error_msg[] = 'area is required';
    }
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
        'companyname' => $companyname,
        'email' => $email,
        'phone' => $phone,
        'city' => $city,
        'area' => $area,
        'psw1' => $psw1,
        'psw2' => $psw2 
    ];

    if(empty($error_msg)){
    	$sql=' INSERT INTO company (`companyname`, `email`, `phone`, `city`, `area`, `password`) VALUES (?, ?, ?, ?, ?, ?)';
    	$stmt=$conn->prepare($sql);
       $stmt->execute(array($companyname, $email,  $phone, $city, $area,$psw1));
       header('Location: login.php');
     
     exit();
    
    }
    else{

        $_SESSION['errors'] = $error_msg;
        $_SESSION['data'] = $data;
        header('Location: sign-up.php');
        exit();
    }







    }
 ?>
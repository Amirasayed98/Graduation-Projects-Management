<?php   
 //load_data_select.php  
 $connect = mysqli_connect("localhost", "root", "", "graduation");
session_start();


//$_SESSION['view'] = $pid;

 function fill_requriments($connect)  
 {  
      $output = '';  
      $sql = "SELECT * FROM company_requriments";  
      $result = mysqli_query($connect, $sql);  
      while($row = mysqli_fetch_array($result))  
      {  
        $id= $row['c_id'];
        $re_id=$row['re_id'];
         $sql2 = "SELECT * FROM company where company_id=$id";  
         $res = mysqli_query($connect, $sql2);  
         while($r = mysqli_fetch_array($res)){
         
          $output .= '
           <form method="post" action="com_more.php">
            <div class="container col-md-12 shadow-lg cont">
            <div class="container-fluid require_head">
                <h4 class="company">Company-Name  : </span> '. $row['comanyname'] .'</h4>
                <p class="city"><span id="city"> City : </span> '. $r['city'] .' </p>
                <p class="city"><span id="area"> Area : </span> '. $r['area'] .' </p>
            </div>

            <div class=" require_body">
                <h3 class="h_body" id="tracks_text"><span id="tracks">Demand Track : </span> '. $row['track'] .'</h3>
                <p class="par" id="description_text"><span id="descreptin"> Idea Description : </span> '. $row['idea'] .' </p>
              
            </div>

            <div class="container-fluid require_head">
                 <button type="submit" class="btn btn-m1 shadow-lg" name="submit" >More</button>
                <input type="hidden" name="viewc" value="'. $row['c_id'].'">
                <input type="hidden" name="vier" value="'. $row['re_id'].'">
            </div>
        
        
        </div>
         
         </form>
  
        ' ; 
        }  
      }  
      return $output;  
 }  
 ?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>company requirements</title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    <link rel="stylesheet" href="css/company requirements.css"/>

    
</head>
<body>

    <nav class="navbar ">
        <div class="fixed-top nav-down">
        <!-- logo -->
        <div class="container-fluid row">
            <a class="navbar-brand  col-lg-4 offset-lg-0 col-sm-12 offset-sm-6" href="Home.html">
                <img src="image/orange-logo.png"  class="my-logo d-inline-block align-top" alt="my-logo">
              </a>
        
        <!-- links -->
        <div class="col-lg-7 d-none d-lg-block d-xl-block ">
          <ul class="  nav   links ">
            <li class="nav-item ">
              <a class="nav-link notactive" href="Home.html">HOME</a>
            </li>
            <li class="nav-item ">
              <a class="nav-link notactive" href="access code.html">GRADUATION PROJECT</a>
            </li>
            <li class="nav-item ">
              <a class="nav-link notactive" href="company.html">COMPANY</a>
            </li>
            <li class="nav-item ">
              <a class="nav-link active" href="company-requirements .php" tabindex="-1" aria-disabled="true">COMPANY REQUIREMENTS</a>
            </li>
            <li class="nav-item ">
                <div class="dropdown">
                    <a class="btn other-features dropdown-toggle" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                        Other Features
  
                    </a>
                  
                    <div class="dropdown-menu dropdown-menu-me" aria-labelledby="dropdownMenuLink">
                        <a class="dropdown-item dropdown-item-me" href="admin-access-code.html"> <ul><li> Results Of GP-Checker</li></ul></a>
  
                      <a class="dropdown-item dropdown-item-me" href="check -2-files.html"><ul><li>Check Plagiarism Online</li></ul></a>
                      <a class="dropdown-item dropdown-item-me" href="what-plagiarism.html"> <ul><li> What Is Plagiarism</li></ul></a>
                    </div>
                  </div>           
             </li>
          </ul> 
        </div>     
        </div>
      </div>
      </nav>
      <!--small-nav  -->
      <div class="pos-f-t d-lg-none d-xl-none ">
  
        <nav class="navbar navbar-dark   ">
          
            <button class="navbar-toggler " type="button" data-toggle="collapse" data-target="#navbarToggleExternalContent" aria-controls="navbarToggleExternalContent" aria-expanded="false" aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
            </button>
        </nav>
        <div class="collapse " id="navbarToggleExternalContent" >
          <div class=" small-links p-4">
            <ul >
              <li class="d-block ">
                <a class="notactive" href="Home.html">HOME</a>
              </li>
              <li class=" d-block">
                <a class="notactive " href="access code.html">GRADUATION PROJECT</a>
              </li>
              <li class=" d-block">
                <a class="notactive " href="company.html">COMPANY</a>
              </li>
              <li class=" d-block">
                <a class="active " href="company-requirements .php" tabindex="-1" aria-disabled="true">COMPANY REQUIREMENTS</a>
              </li>
              
            </ul>
  
          </div>
        </div>
      </div>
  

    <div class=" container mt-4 row">
        <h3 class="title">Ideas put forward by companies for implementation</h3>
        <hr class="col-md-12">
        
       <div class="row cards" id="show_product">  
         <?php echo fill_requriments($connect) ; 
         ?>
    </div>
    </div>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="js/jquery-3.4.1.min.js" ></script>
    <script src="js/popper.min.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <script src="multiselect/jquery.multiselect.js"></script>
    <script>
      $('#trackOpt').multiselect({
  columns: 1,
  placeholder:'Track (1 or more)',
  enableFiltering: true,
  buttonWidth:'400px'
      });


  </script>
</body>
</html>